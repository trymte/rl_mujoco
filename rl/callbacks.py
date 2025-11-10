"""Custom callback for saving GIFs during evaluation."""

import logging
import os

import gymnasium as gym
import imageio
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

logger = logging.getLogger(__name__)


class EvalGifCallback(BaseCallback):
    """Callback to save GIF from evaluation episodes."""

    def __init__(
        self,
        eval_env: VecEnv,
        log_dir: str,
        eval_freq: int = 50000,
        n_eval_episodes: int = 10,
        env_name: str = "Humanoid-v5",
        max_episode_steps: int = 1000,
        seed: int = 42,
        verbose: int = 1,
    ):
        """
        Initialize the callback.

        Args:
            eval_env: Evaluation environment
            log_dir: Directory to save GIFs
            eval_freq: Frequency of evaluation (in steps)
            n_eval_episodes: Number of episodes to evaluate
            env_name: Name of the environment
            max_episode_steps: Maximum steps per episode
            seed: Random seed
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.eval_env = eval_env
        self.log_dir = log_dir
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.env_name = env_name
        self.max_episode_steps = max_episode_steps
        self.seed = seed
        self.gif_dir = os.path.join(log_dir, "gifs")
        os.makedirs(self.gif_dir, exist_ok=True)

    def _on_step(self) -> bool:
        """
        Called at each step during training.

        Returns:
            True to continue training, False to stop
        """
        # Don't save GIFs during training - only on final evaluation
        return True

    def _save_eval_gif(self):
        """Save a GIF from one evaluation episode."""
        try:
            gif_env = gym.make(
                self.env_name,
                max_episode_steps=self.max_episode_steps,
                render_mode="rgb_array",
            )
            gif_obs, _ = gif_env.reset(seed=self.seed + 1000)

            frames = []
            done = False
            truncated = False

            while not (done or truncated) and len(frames) < self.max_episode_steps:
                action, _ = self.model.predict(gif_obs, deterministic=True)

                frame = gif_env.render()
                if frame is not None:
                    frames.append(frame)

                gif_obs, _, done, truncated, _ = gif_env.step(action)

            gif_env.close()

            if len(frames) > 0:
                timestamp = self.n_calls
                gif_filename = f"eval_step_{timestamp}.gif"
                gif_path = os.path.join(self.gif_dir, gif_filename)
                imageio.mimsave(gif_path, frames, fps=30, loop=0)
                if self.verbose > 0:
                    logger.info(f"Saved evaluation GIF to: {gif_path}")

        except Exception as e:
            if self.verbose > 0:
                logger.warning(f"Failed to save evaluation GIF: {e}")
