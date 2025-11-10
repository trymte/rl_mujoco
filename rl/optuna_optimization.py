"""Optuna hyperparameter optimization for RL training."""

import hashlib
import logging
import os
import pathlib
import traceback
from datetime import datetime
from pprint import pformat
from typing import Any, Callable, Dict, Optional, Type

import gymnasium as gym
import imageio
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecEnv

import optuna
import wandb
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from rl.utils import make_vec_env, setup_callbacks

logger = logging.getLogger(__name__)
# wandb might cause an error without this
os.environ.setdefault("WANDB_START_METHOD", "thread")


class OptunaCallback(EvalCallback):
    """Callback for Optuna to report intermediate values and prune trials."""

    def __init__(
        self,
        trial: optuna.Trial,
        eval_env: VecEnv,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        verbose: int = 0,
        use_wandb: bool = False,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=True,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False
        self.use_wandb = use_wandb

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()

            mean_reward = None
            if (
                hasattr(self, "last_mean_reward")
                and self.last_mean_reward is not None
                and hasattr(self.last_mean_reward, "__len__")
                and len(self.last_mean_reward) > 0
            ):
                mean_reward = self.last_mean_reward[-1]
            elif hasattr(self, "last_mean_reward") and isinstance(
                self.last_mean_reward, (int, float)
            ):
                mean_reward = float(self.last_mean_reward)

            if mean_reward is not None:
                self.trial.report(mean_reward, self.eval_idx)

                if self.use_wandb and wandb.run is not None:
                    wandb.log(
                        {
                            "eval/mean_reward": mean_reward,
                            "eval/step": self.eval_idx,
                            "train/total_timesteps": self.n_calls,
                        },
                        step=self.eval_idx,
                    )

                self.eval_idx += 1

                if self.trial.should_prune():
                    self.is_pruned = True
                    if self.use_wandb and wandb.run is not None:
                        wandb.run.summary["state"] = "pruned"
                        wandb.run.summary["final_reward"] = mean_reward
                    return False

        return True


class RLHyperParamOptimizer:
    """Hyperparameter optimizer for RL training using Optuna.

    This class is algorithm-agnostic and can optimize hyperparameters for any
    stable-baselines3 algorithm (PPO, SAC, TD3, etc.).
    """

    def __init__(
        self,
        algorithm: Type[BaseAlgorithm],
        env_name: str,
        hyperparameter_sampler: Callable[[optuna.Trial], Dict[str, Any]],
        n_timesteps: int = 5000000,
        n_eval_episodes: int = 10,
        eval_freq: int = 100000,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        load_if_exists: bool = False,
        verbose: int = 1,
        seed: int = 42,
        device: str = "auto",
        log_dir: str = "./logs",
        max_episode_steps: Optional[int] = None,
        policy: str = "MlpPolicy",
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
    ):
        """
        Initialize the hyperparameter optimizer.

        Args:
            algorithm: The RL algorithm class to optimize (e.g., PPO, SAC, TD3)
            env_name: Name of the gymnasium environment
            hyperparameter_sampler: Function that takes an Optuna trial and returns
                a dictionary of hyperparameters for the algorithm
            n_timesteps: Number of training timesteps per trial
            n_eval_episodes: Number of episodes for evaluation
            eval_freq: Frequency of evaluation during training
            study_name: Name of the Optuna study
            storage: Storage URL for Optuna study (e.g., "sqlite:///optuna.db")
            load_if_exists: Whether to load existing study if it exists
            verbose: Verbosity level
            seed: Random seed
            device: Device to use for training
            log_dir: Directory for logs
            max_episode_steps: Maximum steps per episode
            policy: Policy type (e.g., "MlpPolicy", "CnnPolicy")
            wandb_project: Weights & Biases project name (None to disable)
            wandb_entity: Weights & Biases entity/team name
        """
        self.algorithm = algorithm
        self.env_name = env_name
        self.hyperparameter_sampler = hyperparameter_sampler
        self.n_timesteps = n_timesteps
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.verbose = verbose
        self.seed = seed
        self.device = device
        self.log_dir = log_dir
        self.max_episode_steps = max_episode_steps
        self.policy = policy
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.use_wandb = wandb_project is not None

        if study_name is None:
            algorithm_name = algorithm.__name__.lower()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            hash = hashlib.sha256(
                f"{algorithm_name}_{env_name}_{timestamp}_{seed}".encode()
            ).hexdigest()[:6]
            study_name = f"{algorithm_name}_{env_name}_{timestamp}_{seed}_{hash}"
        if storage is None:
            storage_path = pathlib.Path("optuna/optuna_storage.db").resolve()
            storage_path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
            storage = f"sqlite:///{str(storage_path.absolute())}"
        self.study_name = study_name

        sampler = TPESampler(seed=seed)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2)

        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=load_if_exists,
            direction="maximize",  # Maximize mean reward
            sampler=sampler,
            pruner=pruner,
        )

    def _create_model(
        self, train_env: VecEnv, model_kwargs: Dict[str, Any]
    ) -> BaseAlgorithm:
        """
        Create an RL model with the given hyperparameters.

        Args:
            trial: Optuna trial object
            train_env: Training environment
            hyperparams: Dictionary of hyperparameters

        Returns:
            Initialized RL model
        """
        if self.verbose > 0:
            logger.info("Creating model with hyperparameters:")
            logger.info(pformat(model_kwargs))

        model = self.algorithm(
            policy=self.policy,
            env=train_env,
            tensorboard_log=None,  # Disable tensorboard for trials
            device=self.device,
            verbose=0,  # Disable verbose output for trials
            **model_kwargs,
        )

        return model

    def _evaluate_model(
        self,
        model: BaseAlgorithm,
        eval_env: VecEnv,
        save_gif: bool = False,
        log_dir: Optional[str] = None,
    ) -> float:
        """
        Evaluate a trained model and return mean reward.

        Args:
            model: Trained RL model
            eval_env: Evaluation environment
            save_gif: Whether to save a GIF from the first evaluation episode
            log_dir: Directory to save GIF (if None, uses self.log_dir)

        Returns:
            Mean reward over evaluation episodes
        """

        episode_rewards = []
        obs = eval_env.reset()

        gif_frames = []
        gif_env = None
        gif_episode_reward = None

        if save_gif:
            gif_env = gym.make(
                self.env_name,
                max_episode_steps=self.max_episode_steps,
                render_mode="rgb_array",
            )
            gif_obs, _ = gif_env.reset(seed=self.seed + 1000)

        for episode_idx in range(self.n_eval_episodes):
            episode_reward = 0
            done = False
            step_count = 0
            max_steps = self.max_episode_steps if self.max_episode_steps else 1000

            while not done and step_count < max_steps:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                episode_reward += reward[0] if hasattr(reward, "__len__") else reward
                step_count += 1

                if save_gif and episode_idx == 0 and gif_env is not None:
                    frame = gif_env.render()
                    if frame is not None:
                        gif_frames.append(frame)
                    gif_obs, gif_reward, gif_terminated, gif_truncated, gif_info = (
                        gif_env.step(
                            action.reshape(-1) if hasattr(action, "reshape") else action
                        )
                    )
                    if gif_terminated or gif_truncated:
                        break

            episode_rewards.append(episode_reward)
            if episode_idx == 0:
                gif_episode_reward = episode_reward
            obs = eval_env.reset()

        if save_gif and gif_env is not None:
            gif_env.close()
            if len(gif_frames) > 0:
                try:
                    save_dir = log_dir if log_dir is not None else self.log_dir
                    gif_dir = os.path.join(save_dir, "gifs")
                    os.makedirs(gif_dir, exist_ok=True)
                    gif_filename = f"final_eval_reward_{gif_episode_reward:.2f}.gif"
                    gif_path = os.path.join(gif_dir, gif_filename)
                    imageio.mimsave(gif_path, gif_frames, fps=30, loop=0)
                    if self.verbose > 0:
                        logger.info(f"✓ Saved final evaluation GIF to: {gif_path}")
                except Exception as e:
                    if self.verbose > 0:
                        logger.info(
                            f"Warning: Failed to save final evaluation GIF: {e}"
                        )

        return float(np.mean(episode_rewards))

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.

        Args:
            trial: Optuna trial object

        Returns:
            Mean evaluation reward (to maximize)
        """
        model_kwargs = self.hyperparameter_sampler(trial)

        if self.use_wandb:
            try:
                config = dict(trial.params)
                config["trial.number"] = trial.number
                config["algorithm"] = self.algorithm.__name__
                config["env_name"] = self.env_name
                config["n_timesteps"] = self.n_timesteps
                config["n_eval_episodes"] = self.n_eval_episodes
                config["eval_freq"] = self.eval_freq
                config["seed"] = self.seed

                wandb.init(
                    project=self.wandb_project,
                    entity=self.wandb_entity,
                    config=config,
                    group=self.study_name,
                    name=f"trial_{trial.number}",
                )
            except Exception as e:
                if self.verbose > 0:
                    logger.info(
                        f"Warning: Failed to initialize wandb for trial {trial.number}: {e}"
                    )
                    logger.info("Continuing without wandb logging...")
                self.use_wandb = False

        trial_log_dir = os.path.join(self.log_dir, f"optuna_trial_{trial.number}")

        logger.info(f"\n{'='*60}")
        logger.info(f"🔍 Starting Trial {trial.number}")
        logger.info(f"{'='*60}")

        try:
            train_env = make_vec_env(
                env_name=self.env_name,
                n_envs=1,
                seed=self.seed,
                max_episode_steps=self.max_episode_steps,
                monitor_dir=os.path.join(trial_log_dir, "train_monitor"),
            )

            eval_env = make_vec_env(
                env_name=self.env_name,
                n_envs=1,
                seed=self.seed + 1000,
                max_episode_steps=self.max_episode_steps,
                monitor_dir=os.path.join(trial_log_dir, "eval_monitor"),
            )

            model = self._create_model(train_env, model_kwargs=model_kwargs.copy())

            optuna_callback = OptunaCallback(
                trial=trial,
                eval_env=eval_env,
                n_eval_episodes=self.n_eval_episodes,
                eval_freq=self.eval_freq,
                verbose=self.verbose,
                use_wandb=self.use_wandb,
            )

            callbacks = setup_callbacks(
                eval_env=eval_env,
                log_dir=trial_log_dir,
                eval_freq=self.eval_freq,
                eval_episodes=self.n_eval_episodes,
                save_freq=100000,  # Default save frequency
                env_name=self.env_name,
                max_episode_steps=self.max_episode_steps,
                seed=self.seed,
            )

            callbacks.insert(0, optuna_callback)

            logger.info(
                f"Trial {trial.number}: Starting training for {self.n_timesteps:,} timesteps..."
            )
            model.learn(
                total_timesteps=self.n_timesteps,
                callback=callbacks,
                progress_bar=False,  # Disable progress bar for cleaner output
            )
            logger.info(f"Trial {trial.number}: Training completed!")

            if optuna_callback.is_pruned:
                logger.info(f"Trial {trial.number}: Pruned by Optuna")
                if self.use_wandb and wandb.run is not None:
                    try:
                        wandb.run.summary["state"] = "pruned"
                        wandb.finish(quiet=True)
                    except Exception:
                        pass
                raise optuna.TrialPruned()

            logger.info(f"Trial {trial.number}: Running final evaluation...")
            mean_reward = self._evaluate_model(
                model, eval_env, save_gif=True, log_dir=trial_log_dir
            )
            logger.info(
                f"Trial {trial.number}: Final evaluation reward: {mean_reward:.2f}"
            )

            if self.use_wandb and wandb.run is not None:
                try:
                    wandb.run.summary["final_reward"] = mean_reward
                    wandb.run.summary["state"] = "completed"
                    wandb.finish(quiet=True)
                except Exception:
                    pass

            train_env.close()
            eval_env.close()
            del model, train_env, eval_env

            return mean_reward

        except optuna.TrialPruned:
            if self.use_wandb and wandb.run is not None:
                try:
                    wandb.run.summary["state"] = "pruned"
                    wandb.finish(quiet=True)
                except Exception:
                    pass
            raise
        except Exception as e:
            # If training fails, return a very low reward

            logger.info(f"Trial {trial.number} FAILED with error: {e}")
            logger.info(f"{'='*60}")
            logger.info("Traceback:")
            traceback.print_exc()
            logger.info(f"{'='*60}\n")

            if self.use_wandb and wandb.run is not None:
                wandb.run.summary["state"] = "failed"
                wandb.run.summary["error"] = str(e)
                wandb.finish(quiet=True)
            return float("-inf")

    def optimize(self, n_trials: int = 50) -> optuna.Study:
        """
        Run hyperparameter optimization.

        Args:
            n_trials: Number of optimization trials

        Returns:
            Optuna study object
        """
        logger.info("=" * 60)
        logger.info(f"Starting Optuna optimization: {self.study_name}")
        logger.info(f"Algorithm: {self.algorithm.__name__}")
        logger.info(f"Number of trials: {n_trials}")
        logger.info(f"Timesteps per trial: {self.n_timesteps:,}")
        logger.info(f"Environment: {self.env_name}")
        logger.info("=" * 60)

        self.study.optimize(
            self._objective,
            n_trials=n_trials,
            show_progress_bar=True,
        )

        self._print_results()
        return self.study

    def _print_results(self) -> None:
        logger.info("\n" + "=" * 60)
        logger.info("Optimization finished!")
        logger.info("=" * 60)
        logger.info(f"Number of finished trials: {len(self.study.trials)}")
        logger.info("Best trial:")
        trial = self.study.best_trial
        logger.info(f"  Value (mean reward): {trial.value:.2f}")
        logger.info("  Params:")
        for key, value in trial.params.items():
            logger.info(f"    {key}: {value}")

    def get_best_params(self) -> Dict[str, Any]:
        """
        Get the best hyperparameters found during optimization.

        Returns:
            Dictionary of best hyperparameters
        """
        return self.study.best_trial.params

    def get_best_value(self) -> float:
        """
        Get the best value (mean reward) found during optimization.

        Returns:
            Best mean reward value
        """
        return self.study.best_trial.value
