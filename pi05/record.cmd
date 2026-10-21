@echo off
rem Example capture settings: change IDs, ports, camera indexes before use.
if not defined FOLLOWER_PORT set "FOLLOWER_PORT=COM3"
if not defined LEADER_PORT set "LEADER_PORT=COM4"
if not defined FOLLOWER_ID set "FOLLOWER_ID=my_follower"
if not defined LEADER_ID set "LEADER_ID=my_leader"
if not defined FRONT_CAMERA set "FRONT_CAMERA=0"
if not defined TOP_CAMERA set "TOP_CAMERA=1"
if not defined PI05_DATASET_ROOT set "PI05_DATASET_ROOT=%~dp0..\datasets\so101_dual_new"
lerobot-record ^
  --robot.type=so101_follower ^
  --robot.port=%FOLLOWER_PORT% ^
  --robot.id=%FOLLOWER_ID% ^
  --robot.use_degrees=true ^
  --robot.cameras="{front: {type: opencv, index_or_path: %FRONT_CAMERA%, width: 640, height: 480, fps: 15}, top: {type: opencv, index_or_path: %TOP_CAMERA%, width: 640, height: 480, fps: 15}}" ^
  --teleop.type=so101_leader ^
  --teleop.port=%LEADER_PORT% ^
  --teleop.id=%LEADER_ID% ^
  --teleop.use_degrees=true ^
  --display_data=true ^
  --play_sounds=false ^
  --dataset.repo_id=local/so101_dual_new ^
  --dataset.root="%PI05_DATASET_ROOT%" ^
  --dataset.no_stamp=true ^
  --dataset.single_task="Pick up the orange earplugs in transparent plastic bags from the tabletop one at a time and place them into the bowl on the table until all of them have been transferred." ^
  --dataset.num_episodes=50 ^
  --dataset.episode_time_s=60 ^
  --dataset.reset_time_s=15 ^
  --dataset.fps=15 ^
  --dataset.push_to_hub=false
