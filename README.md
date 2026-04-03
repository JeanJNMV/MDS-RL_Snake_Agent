# Sneakie: Learning to Play the Snake Game in All Its Forms


## Overview

Sneakie trains and evaluates Deep Q-Network (DQN) agents on a custom Snake environment featuring multiple food types, static and dynamic obstacles, and rich reward-shaping options. The goal is to compare how well different agent architectures and DQN improvements generalise as the environment grows in complexity, from a plain 10×10 grid with a single food item all the way to a grid packed with silver food, poison, and moving obstacle walls.

## Key Features

- **Three agent architectures** — MLP (15-feature hand-crafted state), local-window CNN, and full-grid CNN
- **DQN improvements** — Double DQN, Dueling networks, frame stacking, soft/hard target updates, gradient clipping
- **Rich environment** — Gold / silver / poison food, static random obstacles, moving obstacle walls, potential-based reward shaping (distance + body-proximity)
- **Visualisation** — Renders grid observations and saves gameplay GIFs / videos

## Project Structure

```
src/rl_snake/
├── env.py        # Custom Snake environment (SnakeEnv)
├── agent.py      # DQNAgent, CNNDQNAgent, state extractors
├── train.py      # Training entry-point (argparse CLI)
├── evaluate.py   # Cross-environment evaluation with JSON output
└── visuals.py    # Matplotlib renderer + video export

scripts/          # Experiment launch scripts (exp1 – exp9)
notebooks/        # Tutorial.ipynb — interactive walkthrough
run_slurm.sh      # Generic SLURM job submission wrapper
```

## Getting Started

### Requirements

- Python ≥ 3.10
- [`uv`](https://github.com/astral-sh/uv) (recommended) **or** `pip`

### Installation

```bash
# Clone the repository
git clone https://github.com/JeanJNMV/MDS-RL_SnakeAgent.git
cd MDS-RL_Snake_Agent

# Install with uv (creates a virtual environment automatically)
uv sync

# Or with pip
pip install -e .
```

### Training an agent

```bash
# Train the MLP agent on a basic 10×10 grid for 2 000 episodes
uv run src/rl_snake/train.py \
  --agent-type mlp \
  --episodes 2000 --height 10 --width 10 \
  --gold-reward 1.0 --death-reward -1.0 \
  --seed 42 --save-dir checkpoints/run1

# Train a CNN agent with Double DQN and body-proximity shaping
uv run src/rl_snake/train.py \
  --agent-type cnn \
  --double-dqn --dueling \
  --body-proximity-scale 0.1 \
  --episodes 5000 --height 10 --width 10 \
  --save-dir checkpoints/run2
```

Checkpoints are saved under `--save-dir` every `--save-every` episodes. Pass `--save-video` to also write a gameplay GIF alongside each checkpoint.

### Evaluating a checkpoint

```bash
uv run src/rl_snake/evaluate.py \
  --checkpoint checkpoints/run1/ep02000.pt \
  --agent-type mlp \
  --episodes-per-config 100 \
  --height 10 --width 10 \
  --output-json logs/results.json
```

The evaluator runs the agent on all six built-in test configurations (baseline, more food, poison, random obstacles, moving obstacles, full hard) and writes a JSON summary.

### Interactive tutorial

A step-by-step walkthrough is available in [`notebooks/Tutorial.ipynb`](notebooks/Tutorial.ipynb).



## Authors

- **Adonis Jamal** — CentraleSupélec
- **Fotios Kapotos** — CentraleSupélec
- **Jean-Vincent Martini** — CentraleSupélec