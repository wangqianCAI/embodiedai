@echo off
call "%~dp0settings.cmd"
lerobot-record ^
  --robot.type=so101_follower ^
  --robot.port=%FOLLOWER_PORT% ^
  --robot.id=%FOLLOWER_ID% ^
  --robot.cameras="{front: {type: opencv, index_or_path: %FRONT_CAMERA%, width: 640, height: 480, fps: 30, fourcc: MJPG}}" ^
  --teleop.type=so101_leader ^
  --teleop.port=%LEADER_PORT% ^
  --teleop.id=%LEADER_ID% ^
  --display_data=true ^
  --play_sounds=false ^
  --dataset.repo_id=local/so101_test_001 ^
  --dataset.root="%DATASET_ROOT%" ^
  --dataset.no_stamp=true ^
  --dataset.single_task="Pick up the orange earplugs in transparent plastic packaging and place them in the ceramic bowl." ^
  --dataset.num_episodes=30 ^
  --dataset.episode_time_s=20 ^
  --dataset.reset_time_s=10 ^
  --dataset.fps=30 ^
  --dataset.push_to_hub=false
