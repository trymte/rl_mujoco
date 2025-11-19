"""Configuration settings for RL training pipeline."""

from dataclasses import dataclass, field
from typing import Optional

import torch as th


@dataclass
class TrainingConfig:
    """A hardcoded configuration for PPO RL training."""

    env_name: str = "Humanoid-v5"
    max_episode_steps: int = 1000

    total_timesteps: int = 5_000_000
    learning_rate: float = 3e-5
    learning_rate_schedule: str = "constant"  # rl_zoo3 uses constant for Humanoid
    n_steps: int = 512
    batch_size: int = 256
    n_epochs: int = 5
    gamma: float = 0.95
    gae_lambda: float = 0.9
    clip_range: float = 0.2
    clip_range_schedule: str = "constant"
    ent_coef: float = 0.002
    vf_coef: float = 0.45
    max_grad_norm: float = 2.0
    normalize_advantage: bool = True  # Normalize advantages for stability

    # Network architectures
    policy_kwargs: dict = field(
        default_factory=lambda: {
            "net_arch": [dict(pi=[256, 256], vf=[256, 256])],
            "activation_fn": th.nn.ReLU,
        }
    )

    # Logging and checkpointing
    log_dir: str = "./logs"
    tensorboard_log: Optional[str] = "./tensorboard"
    wandb_project: Optional[str] = "humanoid-locomotion"
    wandb_entity: Optional[str] = None
    save_freq: int = 100000
    eval_freq: int = 200000
    eval_episodes: int = 10

    device: str = "auto"  # "auto", "cpu", "cuda"

    seed: int = 42

    checkpoint_path: Optional[str] = None
