#!/bin/bash
# Train the CNN agent on the hard environment with best-practice hyperparameters.
# Key upgrades over the baseline:
#   --n-frames 4      : frame stacking lets the network infer obstacle direction
#   --double-dqn      : reduces Q-value overestimation in complex envs
#   --dueling         : faster value learning when many actions are equivalent
#   --target-tau 0.005: soft Polyak target update — smoother convergence than hard update
#   --grad-clip 10    : gradient clipping — prevents large spikes from outlier Q targets
#   --step-reward     : small negative reward discourages stalling / looping
#   slower epsilon decay + larger buffer : needed for the larger state space
#
# Usage: bash scripts/train_hard_cnn.sh

set -euo pipefail
cd "$(dirname "$0")/.."

uv run src/rl_snake/train.py \
  --agent-type   cnn     \
  --episodes     10000   \
  --max-steps    500     \
  --n-gold       1       \
  --n-silver     1       \
  --n-poison     1       \
  --n-dynamic-obstacles 2 \
  --gold-reward   10.0   \
  --silver-reward  5.0   \
  --poison-reward -5.0   \
  --death-reward  -10.0  \
  --step-reward   -0.01  \
  --n-frames     4       \
  --double-dqn           \
  --dueling              \
  --lr           3e-4    \
  --gamma        0.99    \
  --epsilon-start  1.0   \
  --epsilon-end    0.05  \
  --epsilon-decay  0.9995 \
  --batch-size   128     \
  --buffer-capacity 200000 \
  --target-update  2000  \
  --target-tau     0.005 \
  --grad-clip      10.0  \
  --save-every   1000    \
  --save-video           \
  --save-dir     checkpoints \
  --wandb-project rl-snake   \
  --run-name     hard_cnn_best \
  "$@"
