# Diagnostic Test Commands

Run from `asyncDualPlayPPO/` directory.

## Test 1 — Reward Pipeline

```bash
python train_curobo.py --num_envs 16 --max_iterations 50 \
  --exp_name "diag_N/t1_reward" --save_interval 0 \
  --test_reward_pipeline
```

## Test 2 — Alice Exploration Sandbox

```bash
python train_curobo.py --num_envs 32 --max_iterations 200 \
  --exp_name "diag_N/t2_sandbox" --alice_sandbox
```

Offline analysis:
```bash
python -m asyncDualPlayPPO.diagnostics.test_alice_sandbox \
  --log_dir "runs/diag_N/t2_sandbox/summary"
```

## Test 3 — Full PPO/ABC (50 iters)

```bash
python train_curobo.py --num_envs 32 --max_iterations 50 \
  --exp_name "diag_N/t3_ppo_abc" --save_interval 50
```

Offline analyses:
```bash
python -m asyncDualPlayPPO.diagnostics.test_ppo_abc_balance \
  --log_dir "runs/diag_N/t3_ppo_abc/summary"

python -m asyncDualPlayPPO.diagnostics.test_checkpoint_chain \
  --log_dir "runs/diag_N/t3_ppo_abc/bob"
```

## Test 4 — GoalEncoder (offline, no sim needed)

Uses the Bob checkpoint from Test 3:

```bash
CKPT="asyncDualPlayPPO/runs/diag_1/test_ppo_abc/bob/model_50.pt"
CFG="asyncDualPlayPPO/cfg/ppo/ppo_continuous.yaml"

python -m asyncDualPlayPPO.diagnostics.test_abc_goal_encoder \
  --ckpt "$CKPT" --cfg "$CFG"

python -m asyncDualPlayPPO.diagnostics.test_goal_encoder_latent \
  --ckpt "$CKPT" --cfg "$CFG" \
  --log_dir "asyncDualPlayPPO/runs/diag_1/test_ppo_abc/summary" \
  --save_plot "/tmp/goal_encoder_tsne.png"
```

## Latest Run Paths

| Test | Exp Name | Log Dir |
|------|----------|---------|
| T1 | `diag_1/test_r_pipeline` | runs/diag_1/test_r_pipeline |
| T2 | `diag_1/test_sandbox` | runs/diag_1/test_sandbox |
| T3 | `diag_1/test_ppo_abc` | `runs/diag_1/test_ppo_abc` |
