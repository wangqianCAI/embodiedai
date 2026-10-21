"""Protocol regression checks. No server, cameras or motors are used."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

spec = importlib.util.spec_from_file_location('remote', Path(__file__).parents[1] / 'pi05/remote_client.py')
remote = importlib.util.module_from_spec(spec)
spec.loader.exec_module(remote)


class Response:
    def __init__(self, value): self.value = value
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def read(self): return json.dumps(self.value).encode()


class ProtocolTests(unittest.TestCase):
    def policy(self):
        config = SimpleNamespace(type='pi05', use_relative_actions=False, action_feature_names=remote.NAMES)
        with patch.object(remote, 'health', return_value={}), patch.dict(os.environ, {'SO101_EXECUTION_HORIZON': '20'}):
            return remote.RemotePI05(config)

    def test_queue_and_reset(self):
        policy = self.policy()
        actions = np.arange(300, dtype=np.float32).reshape(50, 6)
        batch = {'observation.state': torch.zeros(1, 6),
                 'observation.images.front': torch.zeros(1, 3, 480, 640),
                 'observation.images.top': torch.ones(1, 3, 480, 640)}
        result = {'request_id': 1, 'rtc_mode': 'off', 'actions': actions.tolist()}
        with patch.object(remote.HTTP, 'open', return_value=Response(result)) as request:
            for i in range(20):
                self.assertTrue(torch.equal(policy.select_action(batch), torch.tensor(actions[i:i+1])))
            self.assertEqual(request.call_count, 1)
            payload = json.loads(request.call_args.args[0].data)
            self.assertIn('top_png', payload)
            self.assertEqual(payload['protocol'], 'chunk-v1')
        policy.reset()
        self.assertEqual(len(policy.queue), 0)

    def test_wrong_checkpoint_rejected(self):
        report = {'status':'ready','policy':'pi05','protocol':'chunk-v1','rtc_mode':'off',
                  'chunk_size':50,'checkpoint':'wrong'}
        with patch.dict(os.environ, {'SO101_EXPECTED_CHECKPOINT':'selected'}), patch.object(remote.HTTP,'open',return_value=Response(report)):
            with self.assertRaises(RuntimeError): remote.health()

    def test_nonfinite_image_rejected(self):
        with self.assertRaises(ValueError): remote.encode_image(torch.full((3,480,640),float('nan')))


if __name__ == '__main__':
    unittest.main()
