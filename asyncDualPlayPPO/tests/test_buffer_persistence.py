"""
Unit tests for GPUDemonstrationBuffer.save() / .load() persistence.

Does NOT require IsaacSim — runs with plain Python + PyTorch.

    python tests/test_buffer_persistence.py
"""

import os
import sys
import tempfile
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from asyncDualPlayPPO.algorithms.rl.ppo.storage import GPUDemonstrationBuffer


def _fill_buffer(buf: GPUDemonstrationBuffer, n: int):
    """Insert n steps of deterministic fake data."""
    obs = torch.arange(n * buf.observations.shape[1], dtype=torch.float32).view(n, -1)
    states = obs.clone()
    acts = torch.arange(n * buf.actions.shape[1], dtype=torch.float32).view(n, -1)
    rews = torch.arange(n, dtype=torch.float32)
    dones = torch.zeros(n, dtype=torch.uint8)
    vals = torch.arange(n, dtype=torch.float32) * 0.1
    log_probs = torch.arange(n, dtype=torch.float32) * -0.01
    mus = acts.clone() * 0.5
    sigmas = acts.clone() * 0.1 + 0.01
    ret = rews.clone() + 1.0
    adv = rews.clone() - 0.5
    buf.add_trajectory(obs, states, acts, rews, dones, vals, log_probs, mus, sigmas, ret, adv)


def test_save_load_partial():
    """Buffer not yet full: step and data survive a round-trip."""
    obs_shape = (51,)
    act_shape = (6,)
    capacity = 500
    n = 120  # less than capacity

    buf = GPUDemonstrationBuffer(capacity, obs_shape, obs_shape, act_shape)
    _fill_buffer(buf, n)
    assert buf.step == n
    assert not buf.full

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        buf.save(path)

        buf2 = GPUDemonstrationBuffer(capacity, obs_shape, obs_shape, act_shape)
        buf2.load(path)

        assert buf2.step == n, f"step mismatch: {buf2.step} != {n}"
        assert not buf2.full
        assert torch.allclose(buf2.observations[:n], buf.observations[:n]), "observations mismatch"
        assert torch.allclose(buf2.actions[:n], buf.actions[:n]), "actions mismatch"
        assert torch.allclose(buf2.returns[:n], buf.returns[:n]), "returns mismatch"
    finally:
        os.unlink(path)

    print("PASS  test_save_load_partial")


def test_save_load_full_wraparound():
    """Buffer that has wrapped around: full flag and all tensors survive."""
    obs_shape = (51,)
    act_shape = (6,)
    capacity = 100
    n = 150  # forces a wrap

    buf = GPUDemonstrationBuffer(capacity, obs_shape, obs_shape, act_shape)
    _fill_buffer(buf, n)
    assert buf.full, "buffer should be full after overflow"

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        buf.save(path)

        buf2 = GPUDemonstrationBuffer(capacity, obs_shape, obs_shape, act_shape)
        buf2.load(path)

        assert buf2.full
        assert buf2.step == buf.step
        assert torch.allclose(buf2.observations, buf.observations), "observations mismatch"
        assert torch.allclose(buf2.actions, buf.actions), "actions mismatch"
    finally:
        os.unlink(path)

    print("PASS  test_save_load_full_wraparound")


def test_size_property():
    """size property returns correct count before and after fill."""
    buf = GPUDemonstrationBuffer(200, (4,), (4,), (2,))
    assert buf.size == 0
    _fill_buffer(buf, 50)
    assert buf.size == 50
    _fill_buffer(buf, 200)  # overfill
    assert buf.size == 200
    print("PASS  test_size_property")


def test_resume_auto_detect(tmp_path):
    """Simulate the train.py resume path: abc_buffer.pt found next to chkpt_bob."""
    obs_shape = (51,)
    act_shape = (6,)
    capacity = 500

    # Simulate a bob checkpoint directory
    bob_dir = str(tmp_path / "bob")
    os.makedirs(bob_dir, exist_ok=True)
    chkpt_bob = os.path.join(bob_dir, "model_50.pt")
    # Write a dummy .pt so isfile passes (content doesn't matter for this path test)
    torch.save({}, chkpt_bob)

    # Save ABC buffer next to it
    buf_orig = GPUDemonstrationBuffer(capacity, obs_shape, obs_shape, act_shape)
    _fill_buffer(buf_orig, 300)
    buf_orig.save(os.path.join(bob_dir, "abc_buffer.pt"))

    # Reproduce train.py resume logic
    _abc_buf_path = os.path.join(os.path.dirname(chkpt_bob), "abc_buffer.pt")
    assert os.path.isfile(_abc_buf_path), "abc_buffer.pt not found by auto-detect"

    buf_loaded = GPUDemonstrationBuffer(capacity, obs_shape, obs_shape, act_shape)
    buf_loaded.load(_abc_buf_path)

    assert buf_loaded.size == 300
    assert torch.allclose(buf_loaded.observations[:300], buf_orig.observations[:300])

    print("PASS  test_resume_auto_detect")


if __name__ == "__main__":
    import pathlib

    test_save_load_partial()
    test_save_load_full_wraparound()
    test_size_property()
    # tmp_path shim for direct execution
    with tempfile.TemporaryDirectory() as d:
        test_resume_auto_detect(pathlib.Path(d))

    print("\nAll tests passed.")
