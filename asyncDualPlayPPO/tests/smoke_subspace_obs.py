#!/usr/bin/env python3
"""Smoke test for the disc orientation-free subspace invariant.

Two layers:
  1. Transform unit check (torch-only, no Isaac) — verifies the load-bearing
     invariant of `reward_pbrs.dpose_and_zero_yaw`:
       char_length < 1e-6  ⟹  obj-yaw slot == 0 AND goal-yaw slot == 0
                                AND d_pose == 2D position distance
       char_length  > 0    ⟹  yaw slots preserved AND d_pose includes the yaw term
     Run anywhere torch is available:
        isaaclab.sh -p tests/smoke_subspace_obs.py
     (or plain `python tests/smoke_subspace_obs.py` in any torch venv).

  2. End-to-end env check (needs the Isaac stack + disc env) — documented below;
     run a 2-env disc validation and confirm the printed obs slots are 0:
        isaaclab.sh -p tests/validate_push_asp.py \
            --chkpt_bob runs/discF_.../bob/model_best.pt \
            --scene-set disc --char-length 0.0 --num_tests 2 --headless
     then check the per-push logs: obj/goal yaw and rot_err print as 0 on disc.
"""
import sys

_OBS_ROBOT_DIM = 6
_OBS_OBJ_STATE_DIM = 14
_OBS_GOAL_DIM = 6
_OBJ_YAW = _OBS_ROBOT_DIM + 5                      # 11
_GOAL_YAW = _OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM + 5  # 25
_TAIL = _OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM + _OBS_GOAL_DIM  # 26


def main():
    import math
    import torch
    from asyncDualPlayPPO.tasks.utils.reward_pbrs import dpose_and_zero_yaw

    def make_obs():
        o = torch.zeros(2, 28)
        o[:, _OBS_ROBOT_DIM + 0] = torch.tensor([0.10, -0.20])   # obj x
        o[:, _OBS_ROBOT_DIM + 1] = torch.tensor([0.50, 0.55])    # obj y
        o[:, _OBJ_YAW] = torch.tensor([1.30, -0.70])             # obj yaw (live)
        o[:, _OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM + 0] = torch.tensor([0.00, 0.10])   # goal x
        o[:, _OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM + 1] = torch.tensor([0.55, 0.40])   # goal y
        o[:, _GOAL_YAW] = torch.tensor([2.00, -1.50])            # goal yaw (live)
        return o

    fails = []

    # ── char_length == 0 : disc subspace ─────────────────────────────────────
    o = make_obs()
    r = dpose_and_zero_yaw(o.clone(), _OBS_ROBOT_DIM, _OBS_OBJ_STATE_DIM, _OBS_GOAL_DIM, 0.0)
    if not torch.allclose(r[:, _OBJ_YAW], torch.zeros(2)):
        fails.append(f"disc: obj-yaw not zeroed: {r[:, _OBJ_YAW].tolist()}")
    if not torch.allclose(r[:, _GOAL_YAW], torch.zeros(2)):
        fails.append(f"disc: goal-yaw not zeroed: {r[:, _GOAL_YAW].tolist()}")
    src = make_obs()
    dx = src[:, _OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM] - src[:, _OBS_ROBOT_DIM]
    dy = src[:, _OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM + 1] - src[:, _OBS_ROBOT_DIM + 1]
    pos2d = torch.sqrt(dx ** 2 + dy ** 2)
    if not torch.allclose(r[:, _TAIL], pos2d, atol=1e-5):
        fails.append(f"disc: d_pose != 2D pos dist: {r[:, _TAIL].tolist()} vs {pos2d.tolist()}")
    exp_bearing = torch.atan2(dy, dx)
    if not torch.allclose(r[:, _TAIL + 1], exp_bearing, atol=1e-5):
        fails.append(f"disc: bearing wrong: {r[:, _TAIL + 1].tolist()} vs {exp_bearing.tolist()}")

    # ── char_length == 0.07 : T-block keeps live yaw + yaw term in d_pose ────
    o = make_obs()
    r2 = dpose_and_zero_yaw(o.clone(), _OBS_ROBOT_DIM, _OBS_OBJ_STATE_DIM, _OBS_GOAL_DIM, 0.07)
    if torch.allclose(r2[:, _OBJ_YAW], torch.zeros(2)):
        fails.append("tblock: obj-yaw was zeroed (should be preserved)")
    if torch.allclose(r2[:, _GOAL_YAW], torch.zeros(2)):
        fails.append("tblock: goal-yaw was zeroed (should be preserved)")
    if not torch.all(r2[:, _TAIL] > pos2d + 1e-6):
        fails.append(f"tblock: d_pose should exceed 2D pos dist (yaw term): "
                     f"{r2[:, _TAIL].tolist()} vs {pos2d.tolist()}")

    # ── tolerant zero-test: 1e-9 still counts as the disc subspace ───────────
    o = make_obs()
    r3 = dpose_and_zero_yaw(o.clone(), _OBS_ROBOT_DIM, _OBS_OBJ_STATE_DIM, _OBS_GOAL_DIM, 1e-9)
    if not torch.allclose(r3[:, _OBJ_YAW], torch.zeros(2)):
        fails.append("char_length=1e-9 did not zero yaw (float-eq regression)")

    if fails:
        print("[SMOKE][FAIL]")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("[SMOKE][PASS] dpose_and_zero_yaw invariant holds "
          "(disc: yaw zeroed + d_pose=2D pos; T-block: yaw live + yaw term; 1e-9 tolerant).")


if __name__ == "__main__":
    main()
