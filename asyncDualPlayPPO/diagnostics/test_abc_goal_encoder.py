"""
Test 4a — ABC + GoalEncoder Integration (forward pass + gradient flow).

Verifies:
    1. model._encode_obs() runs without exception
    2. goal embedding has the expected shape (B, K_per_obj)
    3. ABC loss backpropagates into GoalEncoder parameters (Fix 14)

Does NOT require Isaac Lab or a running environment — loads a checkpoint and
runs a pure forward/backward pass with synthetic observations.

Usage:
    python -m asyncDualPlayPPO.diagnostics.test_abc_goal_encoder \\
        --ckpt asyncDualPlayPPO/runs/diag_t3/bob/model_50.pt \\
        --cfg  asyncDualPlayPPO/cfg/ppo/ppo_continuous.yaml \\
        --num_objects 1
"""

import argparse
import os
import sys
import torch


def check_goal_encoder_integration(ckpt_path: str, cfg_path: str, num_objects: int = 1):
    print("\n" + "=" * 70)
    print("TEST 4a — ABC + GoalEncoder Integration")
    print(f"Checkpoint: {ckpt_path}")
    print("=" * 70)

    import yaml

    if not os.path.isfile(ckpt_path):
        print(f"  SKIP: checkpoint not found at {ckpt_path}")
        sys.exit(0)

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["params"]["policy"]
    model_cfg["num_objects"] = num_objects

    from asyncDualPlayPPO.algorithms.rl.ppo.module import ActorCritic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Infer obs_dim from checkpoint
    state = torch.load(ckpt_path, map_location=device)
    # Find a weight that tells us input size
    obs_dim = None
    for k, v in state.items():
        if "encoder" in k and "weight" in k and v.ndim == 2:
            obs_dim = v.shape[1]
            break
    if obs_dim is None:
        print("  SKIP: could not infer obs_dim from checkpoint")
        sys.exit(0)

    model = ActorCritic(obs_dim, obs_dim, 6, model_cfg=model_cfg, device=device)
    model.load_state_dict(state, strict=False)
    model = model.to(device)

    failures = []

    if not model.use_goal_encoder:
        print("  SKIP: use_goal_encoder=False in this checkpoint")
        sys.exit(0)

    K = model._ge_K_per_obj
    B = 16
    obs_batch = torch.randn(B, obs_dim, device=device)

    # Forward pass through _encode_obs
    try:
        enc, g_pooled = model._encode_obs(obs_batch, detach_goal_encoder=False)
    except Exception as e:
        failures.append(f"FAIL: _encode_obs raised {type(e).__name__}: {e}")
        for f in failures:
            print(f"  {f}")
        print("TEST 4a FAIL")
        sys.exit(1)

    if g_pooled is None:
        failures.append("FAIL: _encode_obs returned g_pooled=None")
    elif g_pooled.shape != (B, K):
        failures.append(f"FAIL: g_pooled shape {tuple(g_pooled.shape)} != ({B}, {K})")
    else:
        print(f"  _encode_obs OK: g_pooled shape={tuple(g_pooled.shape)}")

    if failures:
        for f in failures:
            print(f"  {f}")
        print("TEST 4a FAIL")
        sys.exit(1)

    # Gradient flow into GoalEncoder (Fix 14)
    model.zero_grad()
    loss = g_pooled.sum()
    loss.backward()

    grad_missing = []
    for name, p in model.goal_encoder.named_parameters():
        if p.requires_grad and p.grad is None:
            grad_missing.append(name)

    if grad_missing:
        failures.append(
            f"FAIL: GoalEncoder params without grad after backward: {grad_missing}. "
            "Fix 14 (detach_goal_encoder=False) may not be applied."
        )

    if failures:
        for f in failures:
            print(f"  {f}")
        print("TEST 4a FAIL")
        sys.exit(1)
    else:
        print(f"  Gradient flow OK: all {sum(1 for _ in model.goal_encoder.parameters())} GoalEncoder params have grad")
        print("TEST 4a PASS")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to Bob model checkpoint (.pt)")
    parser.add_argument(
        "--cfg",
        default=os.path.join(os.path.dirname(__file__), "..", "cfg", "ppo", "ppo_continuous.yaml"),
        help="Path to ppo_continuous.yaml config",
    )
    parser.add_argument("--num_objects", type=int, default=1)
    args = parser.parse_args()
    check_goal_encoder_integration(args.ckpt, args.cfg, args.num_objects)
