#!/bin/bash
# Train the MLP agent on the simple environment (baseline — expected to succeed).
# Demonstrates that a lightweight hand-crafted feature vector is sufficient
# when the game has no moving obstacles, no silver food, and no poison.
#
# Usage: bash scripts/train_simple_mlp.sh

set -euo pipefail
cd "$(dirname "$0")/.."

uv run src/rl_snake/train.py \
  --agent-type   mlp     \
  --episodes     3000    \
  --max-steps    500     \
  --n-gold       1       \
  --gold-reward  10.0    \
  --death-reward -10.0   \
  --step-reward  0.0     \
  --lr           1e-3    \
  --gamma        0.99    \
  --epsilon-start  1.0   \
  --epsilon-end    0.01  \
  --epsilon-decay  0.995 \
  --batch-size   64      \
  --buffer-capacity 100000 \
  --target-update  1000  \
  --save-every   500     \
  --save-dir     checkpoints \
  --wandb-project rl-snake   \
  --run-name     simple_mlp  \
  "$@"
