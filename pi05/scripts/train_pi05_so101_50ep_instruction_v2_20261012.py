import os
"""Three epochs with flushed final accumulation groups and epoch checkpoints."""
import json
import sys
import time
from pathlib import Path
import torch
from epoch_accumulation import EpochAccumulation
import lerobot.scripts.lerobot_train as train
from pi05_offline_processors import install
install(train)

root=Path(os.environ.get('SO101_ROOT', '/data/so101_pi0'))
config=root/'configs/pi05_so101_50ep_instruction_v2_20261012.json'
raw=json.loads(config.read_text())
assert raw['steps']==64614 and raw['save_freq']==21538
assert raw['batch_size']==1 and raw['accelerator']['gradient_accumulation']['steps']==4
boundary=EpochAccumulation(21538,4)
original_update=train.update_policy
original_save=train.save_checkpoint
original_step=torch.optim.AdamW.step
updates=0
start=time.perf_counter()
progress=Path(raw['output_dir']).parent/'pi05_so101_50ep_instruction_v2_20261012_progress.json'
def write_progress(status,**extra):
    record={'status':status,'microsteps_completed':boundary.completed,'optimizer_updates':updates,
            'epoch':boundary.completed/21538,'elapsed_s':time.perf_counter()-start,**extra}
    tmp=progress.with_suffix('.tmp')
    tmp.write_text(json.dumps(record,indent=2))
    tmp.replace(progress)
def adam_step(self,*args,**kwargs):
    global updates
    result=original_step(self,*args,**kwargs)
    updates+=1
    return result
torch.optim.AdamW.step=adam_step
def update(*args,**kwargs):
    accelerator=kwargs['accelerator']
    boundary.before(accelerator)
    value=original_update(*args,**kwargs)
    boundary.after(accelerator)
    if boundary.completed % 100 == 0 or boundary.completed % 21538 == 0:
        write_progress('running',loss=float(args[0].loss.val))
    return value
def save(*args,**kwargs):
    assert boundary.completed % 21538 == 0 and kwargs['accelerator'].sync_gradients
    begin=time.perf_counter()
    result=original_save(*args,**kwargs)
    record={'epoch':boundary.completed//21538,'microstep':boundary.completed,'optimizer_updates':updates,
            'save_s':time.perf_counter()-begin,'checkpoint':str(kwargs['checkpoint_dir'])}
    with (Path(raw['output_dir']).parent/'pi05_so101_50ep_instruction_v2_20261012_checkpoints.jsonl').open('a') as f:
        f.write(json.dumps(record)+'\n')
    write_progress('running',last_checkpoint=record)
    return result
train.update_policy=update
train.save_checkpoint=save
sys.argv=[sys.argv[0],'--config_path='+str(config)]
try:
    train.main()
    assert boundary.completed==64614 and updates==16155
    write_progress('completed')
except BaseException as e:
    write_progress('failed',error=type(e).__name__,message=str(e))
    raise
