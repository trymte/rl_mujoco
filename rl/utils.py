"""Utility functions for RL training."""

import os
from typing import Optional

import gymnasium as gym
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from rl.callbacks import EvalGifCallback


def make_env(
    env_name: str, seed: int = 0, max_episode_steps: Optional[int] = None
) -> gym.Env:
    """
    Create a single environment.

    Args:
        env_name: Name of the gymnasium environment
        seed: Random seed for the environment
        max_episode_steps: Maximum steps per episode (None uses default)

    Returns:
        Gymnasium environment
    """

    def _make_env():
        env = gym.make(env_name, max_episode_steps=max_episode_steps)
        env.reset(seed=seed)
        return env

    return _make_env


def make_vec_env(
    env_name: str,
    n_envs: int = 1,
    seed: int = 0,
    max_episode_steps: Optional[int] = None,
    monitor_dir: Optional[str] = None,
) -> VecEnv:
    """
    Create a vectorized environment.

    Args:
        env_name: Name of the gymnasium environment
        n_envs: Number of parallel environments
        seed: Random seed
        max_episode_steps: Maximum steps per episode
        monitor_dir: Directory for monitor logs

    Returns:
        Vectorized environment
    """

    def _make_env(rank: int):
        def _init():
            env = gym.make(env_name, max_episode_steps=max_episode_steps)
            env.reset(seed=seed + rank)
            if monitor_dir is not None:
                monitor_path = os.path.join(monitor_dir, f"monitor_{rank}")
                os.makedirs(monitor_path, exist_ok=True)
                env = Monitor(env, monitor_path)
            return env

        return _init

    if n_envs == 1:
        return DummyVecEnv([_make_env(0)])
    else:
        return DummyVecEnv([_make_env(i) for i in range(n_envs)])


def setup_callbacks(
    eval_env: VecEnv,
    log_dir: str,
    eval_freq: int = 100000,
    eval_episodes: int = 10,
    save_freq: int = 100000,
    best_model_save_path: Optional[str] = None,
    env_name: str = "Humanoid-v5",
    max_episode_steps: int = 1000,
    seed: int = 42,
    verbose: int = 1,
):
    """
    Setup training callbacks.

    Args:
        eval_env: Environment for evaluation
        log_dir: Directory for logs
        eval_freq: Frequency of evaluation (in steps)
        eval_episodes: Number of episodes for evaluation
        save_freq: Frequency of checkpoint saving (in steps)
        best_model_save_path: Path to save best model
        env_name: Name of the environment
        max_episode_steps: Maximum steps per episode
        seed: Random seed
        verbose: Verbosity level

    Returns:
        List of callbacks
    """
    callbacks = []

    if best_model_save_path is None:
        best_model_save_path = os.path.join(log_dir, "best_model")

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_save_path,
        log_path=os.path.join(log_dir, "eval_logs"),
        eval_freq=eval_freq,
        n_eval_episodes=eval_episodes,
        deterministic=True,
        render=False,
        verbose=verbose,
    )
    callbacks.append(eval_callback)

    gif_callback = EvalGifCallback(
        eval_env=eval_env,
        log_dir=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=eval_episodes,
        env_name=env_name,
        max_episode_steps=max_episode_steps,
        seed=seed,
        verbose=verbose,
    )
    callbacks.append(gif_callback)

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="model",
        verbose=verbose,
    )
    callbacks.append(checkpoint_callback)

    return callbacks


def setup_wandb(config) -> Optional[dict]:
    """
    Setup Weights & Biases logging.

    Args:
        config: Training configuration

    Returns:
        wandb_kwargs dict or None if wandb not configured
    """
    if config.wandb_project is None:
        return None

    wandb_kwargs = {
        "project": config.wandb_project,
        "config": config.__dict__,
        "sync_tensorboard": True,
        "monitor_gym": True,
        "save_code": True,
    }

    if config.wandb_entity is not None:
        wandb_kwargs["entity"] = config.wandb_entity

    return wandb_kwargs
