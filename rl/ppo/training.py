"""Training module for a PPO RL agent."""

import logging
import os
import time
from typing import Optional

import gymnasium as gym
import imageio
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.utils import LinearSchedule
from stable_baselines3.common.vec_env import VecEnv

import wandb
from rl.ppo.config import TrainingConfig
from rl.utils import make_vec_env, setup_callbacks, setup_wandb
from wandb.integration.sb3 import WandbCallback

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Training pipeline for RL agents."""

    def __init__(self, config: TrainingConfig):
        """
        Initialize training pipeline.

        Args:
            config: Training configuration
        """
        self.config = config
        self.model: Optional[PPO] = None
        self.train_env: Optional[VecEnv] = None
        self.eval_env: Optional[VecEnv] = None

        os.makedirs(config.log_dir, exist_ok=True)
        if config.tensorboard_log:
            os.makedirs(config.tensorboard_log, exist_ok=True)

    def setup_environments(self):
        """Setup training and evaluation environments."""
        self.train_env = make_vec_env(
            env_name=self.config.env_name,
            n_envs=1,
            seed=self.config.seed,
            max_episode_steps=self.config.max_episode_steps,
            monitor_dir=os.path.join(self.config.log_dir, "train_monitor"),
        )

        self.eval_env = make_vec_env(
            env_name=self.config.env_name,
            n_envs=1,
            seed=self.config.seed + 1000,  # Different seed for eval
            max_episode_steps=self.config.max_episode_steps,
            monitor_dir=os.path.join(self.config.log_dir, "eval_monitor"),
        )

        logger.info(f"Created environments: {self.config.env_name}")
        logger.info(f"Observation space: {self.train_env.observation_space}")
        logger.info(f"Action space: {self.train_env.action_space}")

    def setup_model(self):
        """Setup the RL model."""
        if self.train_env is None:
            raise RuntimeError("Environments must be setup before model creation")

        if self.config.learning_rate_schedule == "linear":
            learning_rate_fn = LinearSchedule(
                initial_value=self.config.learning_rate,
                end_value=self.config.learning_rate * 0.3,  # Decay to 40% of initial
                end_fraction=1.0,  # Decay over entire training period
            )
        else:
            learning_rate_fn = self.config.learning_rate

        if self.config.clip_range_schedule == "linear":
            clip_range_fn = LinearSchedule(
                initial_value=self.config.clip_range,
                end_value=0.0,  # Decay to 0 (rl_zoo3 default)
                end_fraction=1.0,  # Decay over entire training period
            )
        else:
            clip_range_fn = self.config.clip_range

        if self.config.checkpoint_path and os.path.exists(self.config.checkpoint_path):
            logger.info(f"Loading model from checkpoint: {self.config.checkpoint_path}")
            self.model = PPO.load(
                self.config.checkpoint_path,
                env=self.train_env,
                tensorboard_log=self.config.tensorboard_log,
                device=self.config.device,
            )
        else:
            self.model = PPO(
                policy="MlpPolicy",
                env=self.train_env,
                learning_rate=learning_rate_fn,
                n_steps=self.config.n_steps,
                batch_size=self.config.batch_size,
                n_epochs=self.config.n_epochs,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
                clip_range=clip_range_fn,
                ent_coef=self.config.ent_coef,
                vf_coef=self.config.vf_coef,
                max_grad_norm=self.config.max_grad_norm,
                policy_kwargs=self.config.policy_kwargs,
                tensorboard_log=self.config.tensorboard_log,
                device=self.config.device,
                normalize_advantage=self.config.normalize_advantage,
                verbose=1,
            )

        logger.info("Created PPO model")
        logger.info(f"Config: {self.config}")

    def train(self):
        """Run the training loop."""
        if self.model is None:
            raise RuntimeError("Model must be setup before training")

        callbacks = setup_callbacks(
            eval_env=self.eval_env,
            log_dir=self.config.log_dir,
            eval_freq=self.config.eval_freq,
            eval_episodes=self.config.eval_episodes,
            save_freq=self.config.save_freq,
            env_name=self.config.env_name,
            max_episode_steps=self.config.max_episode_steps,
            seed=self.config.seed,
        )

        wandb_kwargs = setup_wandb(self.config)
        if wandb_kwargs is not None:
            wandb.init(**wandb_kwargs)
            wandb_callback = WandbCallback(
                gradient_save_freq=1000,
                model_save_path=f"{self.config.log_dir}/wandb",
                verbose=2,
            )
            callbacks.append(wandb_callback)
            logger.info("Enabled Weights & Biases logging")

        logger.info("=" * 60)
        logger.info("Starting training...")
        logger.info("=" * 60)

        self.model.learn(
            total_timesteps=self.config.total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )

        final_model_path = os.path.join(self.config.log_dir, "final_model")
        self.model.save(final_model_path)
        logger.info(f"Training completed. Final model saved to: {final_model_path}")

        if wandb_kwargs is not None:
            wandb.finish()

    def evaluate(self, n_episodes: int = 10, save_gif: bool = True):
        """
        Evaluate the trained model.

        Args:
            n_episodes: Number of episodes to evaluate
            save_gif: Whether to save a GIF from the first episode
        """
        if self.model is None:
            raise RuntimeError("Model must be setup before evaluation")

        if self.eval_env is None:
            self.eval_env = make_vec_env(
                env_name=self.config.env_name,
                n_envs=1,
                seed=self.config.seed + 1000,
                max_episode_steps=self.config.max_episode_steps,
            )

        logger.info(f"\nEvaluating model over {n_episodes} episodes...")

        episode_rewards = []
        episode_lengths = []
        gif_frames = []
        gif_episode_reward = None
        gif_env = None

        if save_gif:
            gif_env = gym.make(
                self.config.env_name,
                max_episode_steps=self.config.max_episode_steps,
                render_mode="rgb_array",
            )
            logger.info("Capturing frames for GIF from episode 1...")

        obs = self.eval_env.reset()
        for episode in range(n_episodes):
            episode_reward = 0
            episode_length = 0
            done = False

            if save_gif and episode == 0 and gif_env is not None:
                gif_obs, _ = gif_env.reset(seed=self.config.seed + 1000)

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                episode_reward += reward[0]
                episode_length += 1

                if save_gif and episode == 0 and gif_env is not None:
                    frame = gif_env.render()
                    if frame is not None:
                        gif_frames.append(frame)

                    gif_obs, gif_reward, gif_done, gif_truncated, _ = gif_env.step(
                        action.reshape(-1)
                    )
                    if gif_done or gif_truncated:
                        break

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            if episode == 0:
                gif_episode_reward = episode_reward

            logger.info(
                f"Episode {episode + 1}/{n_episodes}: Reward = {episode_reward:.2f}, Length = {episode_length}"
            )

            obs = self.eval_env.reset()

        if save_gif and gif_env is not None:
            gif_env.close()
            if len(gif_frames) > 0:
                gif_path = self._save_gif(gif_frames, gif_episode_reward)
                logger.info(f"Saved GIF to: {gif_path}")

        logger.info("=" * 60)
        logger.info("Evaluation Results:")
        logger.info("=" * 60)
        if episode_rewards:
            logger.info(
                f"Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}"
            )
            logger.info(
                f"Mean Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}"
            )
            logger.info(f"Min Reward: {np.min(episode_rewards):.2f}")
            logger.info(f"Max Reward: {np.max(episode_rewards):.2f}")
        else:
            logger.info("No episodes completed")
        logger.info("=" * 60)

        return {
            "mean_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
            "std_reward": np.std(episode_rewards) if episode_rewards else 0.0,
            "mean_length": np.mean(episode_lengths) if episode_lengths else 0.0,
            "episode_rewards": episode_rewards,
        }

    def _save_gif(self, frames: list[np.ndarray], episode_reward: float) -> str:
        """
        Save frames as a GIF.

        Args:
            frames: List of frames (numpy ndarrays)
            episode_reward: Reward from the episode

        Returns:
            Path to saved GIF file
        """
        gif_dir = os.path.join(self.config.log_dir, "gifs")
        os.makedirs(gif_dir, exist_ok=True)

        timestamp = int(time.time())
        gif_filename = f"eval_episode_reward_{episode_reward:.2f}_{timestamp}.gif"
        gif_path = os.path.join(gif_dir, gif_filename)

        imageio.mimsave(gif_path, frames, fps=30, loop=0)
        return gif_path
