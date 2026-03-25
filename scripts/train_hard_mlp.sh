#!/bin/bash
# Train the MLP agent on the hard environment (expected to struggle/fail).
# Demonstrates the limitations of hand-crafted features:
#   - silver food and poison require food-type discrimination
#   - moving wall obstacles require temporal reasoning (no frame stack)
#   - the 15-feature vector cannot capture full spatial context
#
# Usage: bash scripts/train_hard_mlp.sh

set -euo pipefail
cd "$(dirname "$0")/.."

uv run src/rl_snake/train.py \
  --agent-type   mlp     \
  --episodes     5000    \
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
  --lr           1e-3    \
  --gamma        0.99    \
  --epsilon-start  1.0   \
  --epsilon-end    0.01  \
  --epsilon-decay  0.995 \
  --batch-size   64      \
  --buffer-capacity 100000 \
  --target-update  1000  \
  --save-every   1000    \
  --save-dir     checkpoints \
  --wandb-project rl-snake   \
  --run-name     hard_mlp    \
  "$@"
