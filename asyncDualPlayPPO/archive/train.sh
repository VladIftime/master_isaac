source /home/vlad/env_isaaclab/bin/activate && python -m asyncDualPlayPPO.train \
  --headless \
  --num_envs 512 \
  --exp_name asp_run1 \
  --max_iterations 1000 \
  --save_interval 50