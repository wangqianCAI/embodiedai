@echo off
call "%~dp0settings.cmd"
lerobot-rollout ^
  --strategy.type=base ^
  --policy.path="%ACT_CHECKPOINT%" ^
  --device=cuda ^
  --robot.type=so101_follower ^
  --robot.port=%FOLLOWER_PORT% ^
  --robot.id=%FOLLOWER_ID% ^
  --robot.use_degrees=true ^
  --robot.cameras="{front: {type: opencv, index_or_path: %FRONT_CAMERA%, width: 640, height: 480, fps: 30, fourcc: MJPG}}" ^
  --task="Pick up the orange earplugs in transparent plastic packaging and place them in the ceramic bowl." ^
  --fps=30 ^
  --duration=60 ^
  --interactive=false ^
  --display_data=true
