# π₀.₅ 数据、训练与远程推理

## 数据

实验使用 50 段、21,538 帧、15 FPS 的 LeRobot v3 数据，包含 `observation.images.front`、`observation.images.top`、`observation.state` 和 `action`。
两路图像均为 640×480；状态与动作为 6 维，顺序为 shoulder_pan、shoulder_lift、elbow_flex、wrist_flex、wrist_roll、gripper。
两个物体与碗在各段开始前作小幅随机化放置，示范连续完成两个耳塞入碗。
新采集可参考 Windows CMD 脚本 `pi05\record.cmd`，先设置自己的端口、标定 ID 和相机编号。脚本的单段 60 秒上限和 15 秒重置时间为发布示例参数，不代表历史示范的精确时长。新采数据总帧数通常不同，需调整训练脚本的固定计数。
任务文字：

> Pick up the orange earplugs in transparent plastic bags from the tabletop one at a time and place them into the bowl on the table until all of them have been transferred.

本版本的训练脚本固定于该数据规模。更换数据集时必须重新计算每轮批次数、梯度累积尾组和 checkpoint 间隔，不可只换数据路径。

## 准备服务器目录

预训练模型来自 `lerobot/pi05_base`（实验 revision `22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0`）。先按模型许可获取权重及 `google/paligemma-3b-pt-224` tokenizer，并准备数据。

```bash
export SO101_ROOT=/path/to/so101_workspace
python pi05/prepare_workspace.py
```

准备脚本将脚本、配置复制到新工作目录并重写路径；重复执行会拒绝覆盖现有文件。
然后放置：

```text
$SO101_ROOT/models/pi05_base/                 完整预训练模型
$SO101_ROOT/models/paligemma_tokenizer/       tokenizer 文件
$SO101_ROOT/datasets/so101_test_001_50ep_instruction_v2_20261012/  数据
```

## 训练

在服务器已配置环境中，选择空闲 GPU：

```bash
CUDA_VISIBLE_DEVICES=0 python "$SO101_ROOT/scripts/train_pi05_so101_50ep_instruction_v2_20261012.py"
# 第一阶段完成 3 轮后，续训至第 7 轮
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 "$SO101_ROOT/scripts/train_pi05_epoch4to7_dual.py"
```

有效 batch 为 4，第一阶段单卡 batch 1 × 累积 4；第二阶段两卡各 batch 1 × 累积 2。
第一阶段 16,155 次参数更新，第二阶段新增 21,540 次，合计 37,695 次。
续训保留原学习率调度，未重新开始 warmup；这是一份实际实验复现配置，并非自动搜索得到的最优方案。

## 服务与本地执行

```bash
export SO101_GPU=0
export SO101_PORT=18084
python "$SO101_ROOT/scripts/server_rtc_disabled.py"
```

服务默认加载第 7 轮 checkpoint，可用 `SO101_MODEL` 指定完整 `pretrained_model` 目录。
服务会读取训练数据 episode 5 预热；此步骤不是验证集评测。RTC 默认关闭，预测 50 步动作块。
另开本地终端建立隧道（替换为自己的 SSH 别名，不在仓库中存放私钥）：

```bash
ssh -N -L 18084:127.0.0.1:18084 YOUR_SERVER_ALIAS
```

`GET /health` 返回服务信息。`POST /predict` 请求为 `protocol=chunk-v1`、`request_id`、6 维 `state`、RGB PNG 的 base64 `image_png` 和 `top_png`；返回 50×6 的绝对动作目标。
仓库提供基于原生 rollout 的同步客户端 `remote_client.py`，默认每次执行前 20 步再重新观测。这是发布整理的适配版，做了离线协议检查，未重新进行硬件测试；不把它视为视频原始客户端的逐字备份。

Windows CMD 示例（先替换端口、标定 ID、两路相机编号）：

```bat
set SO101_EXPECTED_CHECKPOINT=/path/to/so101_workspace/outputs/pi05_so101_50ep_epoch4to7_dual_20261012/checkpoints/075383/pretrained_model
python pi05\remote_client.py --strategy.type=base --policy.path=pi05/remote_metadata --device=cpu --robot.type=so101_follower --robot.port=COM3 --robot.id=my_follower --robot.use_degrees=true --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 15}, top: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 15}}" --fps=15 --duration=60 --interactive=false --display_data=true
```

客户端启动前检查模型类型、协议和期望 checkpoint；图像和状态不在客户端重复归一化，服务器完成预处理及反归一化。
同步客户端等待网络时会暂停动作更新，15 FPS 仅为目标值，不声称隐藏推理延迟或达到严格实时。
