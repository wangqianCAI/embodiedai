"""Synchronous chunk-v1 adapter for native LeRobot rollout, no local policy weights."""
import base64
from collections import deque
import json
import os
import time
import urllib.request

import cv2
import numpy as np
import torch
from torch import nn

URL = os.environ.get('SO101_URL', 'http://127.0.0.1:18084').rstrip('/')
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))
NAMES = ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos',
         'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']


def health():
    expected = os.environ.get('SO101_EXPECTED_CHECKPOINT')
    if not expected:
        raise ValueError('Set SO101_EXPECTED_CHECKPOINT to the server checkpoint path')
    with HTTP.open(URL + '/health', timeout=5) as response:
        report = json.load(response)
    if (report.get('status') != 'ready' or report.get('policy') != 'pi05'
            or report.get('protocol') != 'chunk-v1' or report.get('rtc_mode') != 'off'
            or report.get('chunk_size') != 50 or report.get('checkpoint') != expected):
        raise RuntimeError('Remote service does not match the selected pi05 checkpoint/protocol')
    return report


def encode_image(tensor):
    image = tensor.detach().cpu()
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if tuple(image.shape) != (3, 480, 640) or not torch.isfinite(image).all():
        raise ValueError('Expected finite RGB CHW 640x480 image')
    if image.min() < 0 or image.max() > 1:
        raise ValueError('Expected RGB image scaled to [0,1]')
    rgb = (image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    ok, png = cv2.imencode('.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError('PNG encoding failed')
    return base64.b64encode(png).decode('ascii')


class Identity:
    steps = ()
    def __call__(self, value):
        return value
    def reset(self):
        pass


class RemotePI05(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.type != 'pi05' or config.use_relative_actions or config.action_feature_names != NAMES:
            raise ValueError('Unexpected model, units mode or joint order')
        health()  # Called before rollout connects to hardware.
        self.config = config
        self.type = 'pi05'
        self.queue = deque()
        self.request_id = 0
        self.horizon = int(os.environ.get('SO101_EXECUTION_HORIZON', '20'))
        if not 1 <= self.horizon <= 50:
            raise ValueError('Execution horizon must be in [1,50]')

    def reset(self):
        self.queue.clear()

    def drop_queued_actions(self):
        self.queue.clear()

    def supports_text_generation(self):
        return False

    def select_action(self, batch):
        if not self.queue:
            state = batch['observation.state'].detach().cpu().reshape(-1).tolist()
            if len(state) != 6 or not np.isfinite(state).all():
                raise ValueError('Expected six finite joint values')
            self.request_id += 1
            payload = {'protocol': 'chunk-v1', 'request_id': self.request_id, 'state': state,
                       'image_png': encode_image(batch['observation.images.front']),
                       'top_png': encode_image(batch['observation.images.top'])}
            request = urllib.request.Request(URL + '/predict', data=json.dumps(payload).encode(),
                                             headers={'Content-Type': 'application/json'})
            start = time.monotonic()
            with HTTP.open(request, timeout=5) as response:
                result = json.load(response)
            if time.monotonic() - start > 3:
                raise RuntimeError('Observation-to-action delay exceeded 3 seconds')
            actions = np.asarray(result['actions'], dtype=np.float32)
            if (result.get('request_id') != self.request_id or result.get('rtc_mode') != 'off'
                    or actions.shape != (50, 6) or not np.isfinite(actions).all()):
                raise ValueError('Invalid action response')
            self.queue.extend(torch.from_numpy(actions[:self.horizon].copy()).unsqueeze(1).unbind(0))
        return self.queue.popleft()


def main():
    import lerobot.rollout.context as context
    context._load_pretrained_policy = RemotePI05
    context.make_pre_post_processors = lambda **kwargs: (Identity(), Identity())
    from lerobot.scripts.lerobot_rollout import main as rollout
    rollout()


if __name__ == '__main__':
    main()
