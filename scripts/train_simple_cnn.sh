#!/bin/bash
# Train the CNN agent on the simple environment (baseline — expected to succeed).
# Demonstrates that a full spatial grid observation also works on the easy game,
# giving a fair apples-to-apples comparison against the MLP baseline.
#
# Usage: bash scripts/train_simple_cnn.sh

set -euo pipefail
cd "$(dirname "$0")/.."

uv run src/rl_snake/train.py \
  --agent-type   cnn     \
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
  --run-name     simple_cnn  \
  "$@"
