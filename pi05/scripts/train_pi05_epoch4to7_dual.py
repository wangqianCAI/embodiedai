"""Resume epoch 3 on two GPUs, retain global batch 4, finish epoch 7."""
import os
import sys
import json
import math
import time
from pathlib import Path
import torch
from epoch_accumulation import EpochAccumulation
import lerobot.scripts.lerobot_train as train
from pi05_offline_processors import install
install(train)

ROOT=Path(os.environ.get('SO101_ROOT', '/data/so101_pi0'))
TAG='pi05_so101_50ep_epoch4to7_dual_20261012'
SOURCE=ROOT/'outputs/pi05_so101_50ep_instruction_v2_20261012/checkpoints/064614'
OUT=ROOT/'outputs'/TAG
EPOCH_STEPS=10769
INITIAL=32307
TOTAL=75383
RANK=int(os.environ.get('RANK','0'))
boundary=EpochAccumulation(EPOCH_STEPS,2,initial_microstep=INITIAL)
updates=0
started=time.perf_counter()
timings=[]
original_resume=train.resume_before_prepare
original_metadata=train.load_training_metadata
original_update=train.update_policy
original_save=train.save_checkpoint
original_step=torch.optim.AdamW.step

def resume(cfg):
    step=original_resume(cfg)
    assert step==64614 and Path(cfg.checkpoint_path)==SOURCE
    print(f'RESUME: original 64614 single-GPU batches -> {INITIAL} dual-GPU batches; same 64614 samples',flush=True)
    return INITIAL

def metadata(path):
    data=original_metadata(path)
    if Path(path)==SOURCE/'training_state':
        assert data['dp_world_size']==1 and data['batch_size']==1
        data=dict(data,dp_world_size=2)
    return data

def progress(status,**extra):
    if RANK:return
    mean=sum(timings)/len(timings) if timings else None
    record={'status':status,'gpus':[0,1],'source_checkpoint':str(SOURCE),
            'epoch':boundary.completed/EPOCH_STEPS,'target_epoch':7,'additional_epochs':4,
            'global_microstep':boundary.completed,'new_microsteps':boundary.completed-INITIAL,
            'new_optimizer_updates':updates,'previous_optimizer_updates':16155,
            'elapsed_s':time.perf_counter()-started,'recent_step_s':mean,
            'estimated_remaining_s':(TOTAL-boundary.completed)*mean if mean else None,**extra}
    path=ROOT/'outputs'/f'{TAG}_progress.json'
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2));tmp.replace(path)

def adam_step(self,*args,**kwargs):
    global updates
    result=original_step(self,*args,**kwargs);updates+=1;return result

def update(*args,**kwargs):
    now=time.perf_counter()
    if hasattr(update,'last'):
        timings.append(now-update.last)
        del timings[:-200]
    update.last=now
    accelerator=kwargs['accelerator']
    assert accelerator.num_processes==2
    boundary.before(accelerator)
    value=original_update(*args,**kwargs)
    boundary.after(accelerator)
    loss=float(args[0].loss.val)
    if not math.isfinite(loss):raise RuntimeError(f'Non-finite loss: {loss}')
    if (boundary.completed-INITIAL)%50==0 or boundary.completed%EPOCH_STEPS==0:
        progress('running',loss=loss,lr=float(args[0].lr.val))
    return value

def save(*args,**kwargs):
    assert boundary.completed%EPOCH_STEPS==0 and kwargs['accelerator'].sync_gradients
    begin=time.perf_counter()
    result=original_save(*args,**kwargs)
    if RANK==0:
        entry={'epoch':boundary.completed//EPOCH_STEPS,'checkpoint':str(kwargs['checkpoint_dir']),
               'new_optimizer_updates':updates,'save_s':time.perf_counter()-begin}
        with (ROOT/'outputs'/f'{TAG}_checkpoints.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
        progress('running',last_checkpoint=entry)
    return result

train.resume_before_prepare=resume
train.load_training_metadata=metadata
train.update_policy=update
train.save_checkpoint=save
torch.optim.AdamW.step=adam_step
sys.argv=[sys.argv[0],f'--config_path={SOURCE}/pretrained_model/train_config.json',
          '--resume=true',f'--output_dir={OUT}',f'--job_name={TAG}',f'--steps={TOTAL}',
          f'--save_freq={EPOCH_STEPS}','--log_freq=50','--batch_size=1','--num_workers=0',
          '--parallelism.dp_replicate=2','--parallelism.dp_shard=1',
          '--accelerator.gradient_accumulation.steps=2']
try:
    progress('starting')
    train.main()
    assert boundary.completed==TOTAL and updates==21540,(boundary.completed,updates)
    progress('completed')
except BaseException as exc:
    progress('failed',error=type(exc).__name__,message=str(exc))
    raise
