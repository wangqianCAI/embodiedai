@echo off
call "%~dp0settings.cmd"
lerobot-train ^
  --dataset.repo_id=local/so101_test_001 ^
  --dataset.root="%DATASET_ROOT%" ^
  --dataset.streaming=false ^
  --dataset.video_backend=pyav ^
  --policy.type=act ^
  --policy.device=cuda ^
  --policy.push_to_hub=false ^
  --output_dir="%ACT_OUTPUT%" ^
  --job_name=act_orange_001 ^
  --wandb.enable=false ^
  --steps=20000 ^
  --batch_size=4 ^
  --num_workers=0 ^
  --save_freq=2000 ^
  --log_freq=100
