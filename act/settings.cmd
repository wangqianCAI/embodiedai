@echo off
if not defined DATASET_ROOT set "DATASET_ROOT=%~dp0..\datasets\so101_test_001"
if not defined ACT_OUTPUT set "ACT_OUTPUT=%~dp0..\outputs\act_orange_001"
if not defined ACT_CHECKPOINT set "ACT_CHECKPOINT=%ACT_OUTPUT%\checkpoints\020000\pretrained_model"
if not defined FOLLOWER_PORT set "FOLLOWER_PORT=COM3"
if not defined LEADER_PORT set "LEADER_PORT=COM4"
if not defined FOLLOWER_ID set "FOLLOWER_ID=my_follower"
if not defined LEADER_ID set "LEADER_ID=my_leader"
if not defined FRONT_CAMERA set "FRONT_CAMERA=0"
