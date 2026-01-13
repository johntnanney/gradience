import unittest
import torch
import torch.nn as nn

from gradience.vnext.experimental.guard import LoRAGuard


class TinyLoRAModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Fake "LoRA" params (name contains 'lora')
        self.lora_A = nn.Parameter(torch.randn(4, 8))
        self.lora_B = nn.Parameter(torch.randn(8, 4))
        # Non-LoRA param should not be snapshotted
        self.weight = nn.Parameter(torch.randn(8, 8))

    def named_parameters(self, prefix="", recurse=True):
        for name, p in super().named_parameters(prefix=prefix, recurse=recurse):
            yield name, p


class TestLoRAGuard(unittest.TestCase):
    def test_snapshot_and_rollback_restores(self):
        m = TinyLoRAModel()
        g = LoRAGuard(ring_size=3, cooldown_steps=0)

        original_A = m.lora_A.detach().clone()
        original_B = m.lora_B.detach().clone()

        g.snapshot(10, m, loss=1.0)
        # Corrupt weights
        m.lora_A.data.fill_(999.0)
        m.lora_B.data.fill_(888.0)

        restored = g.rollback(m, steps_back=1)
        self.assertEqual(restored, 10)
        self.assertTrue(torch.allclose(m.lora_A.detach(), original_A))
        self.assertTrue(torch.allclose(m.lora_B.detach(), original_B))

    def test_ring_buffer_eviction(self):
        m = TinyLoRAModel()
        g = LoRAGuard(ring_size=2, cooldown_steps=0)

        g.snapshot(1, m)
        g.snapshot(2, m)
        g.snapshot(3, m)

        self.assertEqual(g.snapshot_count(), 2)
        steps = [s.step for s in list(g.snapshots)]
        self.assertEqual(steps, [2, 3])

    def test_check_triggers(self):
        g = LoRAGuard()

        self.assertEqual(g.check_triggers(loss=float("nan")), "nan_loss")
        self.assertEqual(g.check_triggers(loss=float("inf")), "nan_loss")
        self.assertEqual(g.check_triggers(grad_norm=float("nan")), "nan_grad")
        self.assertEqual(g.check_triggers(grad_norm=1e9), "grad_explosion")

        self.assertIsNone(g.check_triggers(loss=1.0, grad_norm=1.0))

    def test_can_attempt_rollback_cooldown(self):
        m = TinyLoRAModel()
        g = LoRAGuard(cooldown_steps=10, max_rollbacks=10, window_steps=100)

        g.snapshot(10, m)
        g.rollback(m)
        # too soon
        self.assertFalse(g.can_attempt_rollback(15))
        # past cooldown
        self.assertTrue(g.can_attempt_rollback(25))

    def test_can_attempt_rollback_window(self):
        m = TinyLoRAModel()
        g = LoRAGuard(cooldown_steps=0, max_rollbacks=2, window_steps=50)

        # simulate rollbacks at steps 10 and 20
        g.snapshot(10, m); g.rollback(m)
        g.snapshot(20, m); g.rollback(m)

        self.assertFalse(g.can_attempt_rollback(25))  # exceeded max_rollbacks in window
        self.assertTrue(g.can_attempt_rollback(100))  # old rollbacks fall out of window


if __name__ == "__main__":
    unittest.main()
