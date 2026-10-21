"""Published snapshot of the dual-camera chunk-v1 inference server. RTC defaults off."""
from __future__ import annotations

import os

# ---- 必须在导入 torch 之前决定可见 GPU ----
GPU = os.environ.get("SO101_GPU", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
os.environ.update(
    HF_HUB_OFFLINE="1",
    TOKENIZERS_PARALLELISM="false",
    OMP_NUM_THREADS="4",
    NCCL_P2P_DISABLE="1",
    NCCL_IB_DISABLE="1",
)

import base64  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from http.server import BaseHTTPRequestHandler, HTTPServer  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402

ROOT = Path(os.environ.get("SO101_ROOT", "/data/so101_pi0"))
POLICY_TYPE = os.environ.get("SO101_POLICY", "pi05")
MODEL = Path(os.environ.get(
    "SO101_MODEL",
    str(ROOT / "outputs/pi05_so101_50ep_epoch4to7_dual_20261012/checkpoints/075383/pretrained_model"),
))
PORT = int(os.environ.get("SO101_PORT", "18084"))
# 必须与训练时的任务文字完全一致（tokenizer 用它就是这段文本训练的）
TASK = "Pick up the orange earplugs in transparent plastic bags from the tabletop one at a time and place them into the bowl on the table until all of them have been transferred."

print(f"[server_gpu] SO101_GPU={GPU} (物理卡) PORT={PORT} POLICY={POLICY_TYPE}", flush=True)
print(f"[server_gpu] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)

os.chdir(MODEL)
cfg = PreTrainedConfig.from_pretrained(str(MODEL), local_files_only=True)
cfg.device = "cuda"
from lerobot.policies.rtc.configuration_rtc import RTCConfig
RTC_ENABLED = os.environ.get('SO101_RTC_ENABLED', '0') == '1'
RTC_MODE = 'guided' if RTC_ENABLED else 'off'
PROTOCOL = 'rtc-v1' if RTC_ENABLED else 'chunk-v1'
cfg.rtc_config = RTCConfig(enabled=RTC_ENABLED, mode="guided", execution_horizon=20, max_guidance_weight=10.0)
if cfg.type != POLICY_TYPE:
    raise ValueError(f"Checkpoint policy type mismatch: {cfg.type} != {POLICY_TYPE}")

if POLICY_TYPE == "act":
    from lerobot.policies.act.modeling_act import ACTPolicy

    cfg.pretrained_backbone_weights = None
    policy = ACTPolicy(cfg)
elif POLICY_TYPE == "pi05":
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    cfg.gradient_checkpointing = False
    policy = PI05Policy(cfg)
else:
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy

    cfg.gradient_checkpointing = False
    policy = PI0Policy(cfg)

# Fail closed：与上游 from_pretrained 不同，加载出错必须终止服务而不是静默继续
weights = load_file(str(MODEL / "model.safetensors"))
policy.load_state_dict(weights, strict=True)
del weights
policy.to("cuda").eval()

pre, post = make_pre_post_processors(
    policy_cfg=cfg,
    pretrained_path=str(MODEL),
    preprocessor_overrides={"device_processor": {"device": "cuda"}},
    postprocessor_overrides={"device_processor": {"device": "cpu"}},
)

print(f"[server_gpu] 权重加载完成，torch.cuda 当前设备 = {torch.cuda.current_device()}", flush=True)
print(f"[server_gpu] 显存占用 = {torch.cuda.memory_allocated()/1024**3:.2f} GiB", flush=True)


def predict(state, rgb, top, previous=None, delay=0):
    batch = {
        "observation.state": torch.tensor(state, dtype=torch.float32),
        "observation.images.front": torch.from_numpy(np.array(rgb, copy=True)).permute(2, 0, 1).float() / 255,
        "observation.images.top": torch.from_numpy(np.array(top, copy=True)).permute(2, 0, 1).float() / 255,
        "task": TASK,
    }
    start = time.perf_counter()
    prefix = None
    if not RTC_ENABLED and (previous is not None or delay != 0):
        raise ValueError('RTC is disabled: prefixes and delay compensation are not accepted')
    if previous is not None:
        arr = np.asarray(previous, dtype=np.float32)
        if arr.ndim != 2 or not 1 <= len(arr) <= 20 or arr.shape[1] != 6 or not np.isfinite(arr).all():
            raise ValueError('Invalid normalized RTC prefix')
        prefix = torch.tensor(arr, device='cuda')
    if not isinstance(delay, int) or not 0 <= delay < cfg.chunk_size:
        raise ValueError('Invalid RTC delay')
    # Guided RTC needs autograd inside the denoiser: never use inference_mode here.
    with torch.no_grad():
        original = (policy.predict_action_chunk(pre(batch), prev_chunk_left_over=prefix, inference_delay=delay)
                    if RTC_ENABLED else policy.predict_action_chunk(pre(batch)))
        raw = original.detach().cpu().float().numpy().reshape(-1, 6)
        actions = post(original.clone()).cpu().float().numpy()
    torch.cuda.synchronize()
    actions = actions.reshape(-1, 6)
    if not np.isfinite(actions).all() or not np.isfinite(raw).all():
        raise ValueError("Non-finite model output")
    return {"actions": actions.tolist(), "raw_actions": raw.tolist(), "rtc_mode": RTC_MODE,
            "inference_s": time.perf_counter() - start, "prefix_steps": 0 if prefix is None else len(prefix)}


# Use training episode 5 for warmup only; this is NOT held-out evaluation.
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

ds = LeRobotDataset(
    "local/so101_test_001",
    root=ROOT / "datasets/so101_test_001_50ep_instruction_v2_20261012",
    episodes=[5],
    video_backend="pyav",
)
sample = ds[0]
rgb = (sample["observation.images.front"].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
top = (sample["observation.images.top"].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
warmups = [predict(sample["observation.state"].tolist(), rgb, top) for _ in range(3)]
guided_warmup = (predict(sample['observation.state'].tolist(), rgb, top, warmups[-1]['raw_actions'][10:30], 8)
                 if RTC_ENABLED else None)

report = {
    "status": "ready",
    "rtc_mode": RTC_MODE,
    "protocol": PROTOCOL,
    "chunk_size": cfg.chunk_size,
    "execution_horizon": 20,
    "guided_warmup_s": guided_warmup['inference_s'] if guided_warmup else None,
    "policy": POLICY_TYPE,
    "checkpoint": str(MODEL),
    "physical_gpu": int(GPU),
    "torch_device": torch.cuda.current_device(),
    "device_name": torch.cuda.get_device_name(0),
    "port": PORT,
    "warmup_seconds": [x["inference_s"] for x in warmups],
    "allocated_gib": torch.cuda.memory_allocated() / 1024**3,
    "task": TASK,
}
(ROOT / "logs" / f"{POLICY_TYPE}_50ep_gpu{GPU}_ready.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
del ds


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.respond(200, report)

    def do_POST(self):
        try:
            size = int(self.headers["Content-Length"])
            if size > 4000000:
                raise ValueError("Request too large")
            data = json.loads(self.rfile.read(size))
            state = np.asarray(data["state"], dtype=np.float32)
            if state.shape != (6,) or not np.isfinite(state).all():
                raise ValueError("Invalid state (expect 6 finite floats)")
            image = Image.open(io.BytesIO(base64.b64decode(data["image_png"]))).convert("RGB")
            if image.size != (640, 480):
                raise ValueError(f"Expected 640x480 camera, got {image.size}")
            top = Image.open(io.BytesIO(base64.b64decode(data["top_png"]))).convert("RGB")
            if top.size != (640, 480):
                raise ValueError("Top camera must be 640x480")
            if data.get('protocol') != PROTOCOL:
                raise ValueError('Client protocol does not match server execution mode')
            result = predict(state.tolist(), image, top, data.get('previous'), data.get('delay', 0))
            result['request_id'] = data['request_id']
            self.respond(200, result)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self.respond(500, {"error": str(exc)})

    def respond(self, status, data):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *a):  # 降低日志噪音，保留一行摘要
        print(f"{self.address_string()} {fmt % a}", flush=True)


HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
