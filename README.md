# RL-Sandbox

Sandbox for playing around with mujoco, RL, hyperparameter optimization and VLAs++.
Work in progress.

## Installation

Install the project and its dependencies using `uv`:
```bash
uv sync
```

## Basic Training

Train a PPO humanoid locomotion policy for Humanoid-v5 with default settings:
```bash
uv run run_training
```

## Hyperparameter Optimization

Example optimization of PPO hyperparameters on the Humanoid-v5 environment:
```bash
uv run run_optuna --n-trials 100 
```
with 100 optuna trials.

## License

MIT License