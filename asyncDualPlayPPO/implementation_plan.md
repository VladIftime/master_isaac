# Fix Bob's Learning: Update Rate Imbalance + Disable BCO

Bob cannot learn because (1) Alice updates 30–70× faster, shifting the demonstration distribution before Bob can adapt, and (2) the IDM produces garbage actions for out-of-distribution Alice trajectories, poisoning the BC buffer.

## Proposed Changes

### Config

#### [MODIFY] [ppo_continuous.yaml](file:///home3/s3426394/master_isaac/asyncDualPlayPPO/cfg/ppo/ppo_continuous.yaml)

- Set `abc_coef: 0.0` to fully disable BCO/IDM behavioral cloning
- Update comment to reflect the change

---

### Training Loop

#### [MODIFY] [train.py](file:///home3/s3426394/master_isaac/asyncDualPlayPPO/train.py)

**Fix the update imbalance** by capping how many Alice updates can happen between Bob updates:

1. Add a `--max_alice_bob_ratio` CLI argument (default: `5`), controlling the max Alice updates per Bob update.
2. In [perform_alice_update()](file:///home3/s3426394/master_isaac/asyncDualPlayPPO/train.py#174-197), check if `alice_updates - (bob_updates * max_ratio)` exceeds the limit. If so, skip the Alice PPO update (but still buffer the transitions — don't drop data).
3. This ensures Alice progresses at most 5× faster than Bob, preventing catastrophic non-stationarity.

The key logic change in [perform_alice_update()](file:///home3/s3426394/master_isaac/asyncDualPlayPPO/train.py#174-197):
```python
# Gate: don't let Alice outpace Bob too much
if alice_updates >= (bob_updates + 1) * max_alice_bob_ratio:
    return  # Alice waits for Bob to catch up
```

> [!IMPORTANT]
> Alice rollout **data is still collected** even when updates are paused — only the SGD update is skipped. The storage is cleared when it overflows its capacity, so no data is lost but Alice's policy stays frozen until Bob catches up.

**Clean up logging**: When `abc_coef == 0.0`, print `BC=off | IDM=off` instead of the loss values to make logs clearer.

---

## Verification Plan

### Automated Tests
- No automated tests exist for this training loop. A syntax check will confirm no import/parse errors:
  ```
  cd /home3/s3426394/master_isaac/asyncDualPlayPPO && python -c "import ast; ast.parse(open('train.py').read()); print('OK')"
  ```

### Manual Verification
- **Run a short training job** with the fixed code on the HPC cluster and verify:
  1. The log shows Alice updates are being gated (no more than ~5 Alice updates between consecutive Bob updates)
  2. `BC=off | IDM=off` appears in Bob's log lines
  3. Bob's success rate no longer collapses catastrophically
