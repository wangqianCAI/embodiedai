# 安装与源码来源

使用 Python 3.12。先安装与你的 CUDA / 驱动兼容的 PyTorch，再安装项目所需依赖。

ACT 本地实验使用 Seeed LeRobot commit `4aaff99be4a1d81568c08c8f0296b41b40c99ec4`，PyTorch 2.11 + CUDA 12.8。
建议独立环境安装该版本以复现 ACT 的原生 rollout：

```bash
git clone https://github.com/Seeed-Projects/lerobot.git lerobot-act
cd lerobot-act
git checkout 4aaff99be4a1d81568c08c8f0296b41b40c99ec4
pip install -e ".[training,core_scripts,feetech]"
```

π₀.₅ 服务器源码当时以文件快照部署，没有 `.git` 元数据；本仓库保留其实际源码与原许可，不能把它当成经过核验的上游原版提交。
服务器使用的关键包版本见 [版本记录](../evidence/server-versions.txt)。

```bash
# 在本仓库根目录，使用独立环境
pip install -e "./third_party/lerobot[pi,training,core_scripts,feetech]"
```

上面的安装命令不是跨操作系统锁文件。服务器与 Windows 客户端的 PyTorch/CUDA 需要按各自环境安装。
硬件使用前需完成本机 SO-101 标定，保持采集和推理时角度单位、关节顺序、相机位置一致。
不要直接使用其他机械臂的标定参数。首次执行时移开运动路径中的人手并准备停止操作。
