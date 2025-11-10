import argparse
import json
import logging

from stable_baselines3 import PPO, SAC

from rl.hyperparam_sampling import sample_ppo_hyperparameters, sample_sac_params
from rl.optuna_optimization import RLHyperParamOptimizer

logger = logging.getLogger(__name__)


# Algorithm registry: maps algorithm names to (class, sampler function)
ALGORITHM_REGISTRY = {
    "ppo": (PPO, sample_ppo_hyperparameters),
    "sac": (SAC, sample_sac_params),
    # "td3": (TD3, sample_td3_params),
    # "a2c": (A2C, sample_a2c_params),
    # "ddpg": (DDPG, sample_ddpg_params),
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Optuna Hyperparameter Optimization for RL Algorithms"
    )

    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=list(ALGORITHM_REGISTRY.keys()),
        help=f"RL algorithm to optimize (default: ppo, available: {', '.join(ALGORITHM_REGISTRY.keys())})",
    )

    parser.add_argument(
        "--env-name",
        type=str,
        default="Humanoid-v5",
        help="Gymnasium environment name (default: Humanoid-v5)",
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of optimization trials (default: 50)",
    )
    parser.add_argument(
        "--n-timesteps",
        type=int,
        default=5_000_000,
        help="Number of training timesteps per trial (default: 5_000_000)",
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=10,
        help="Number of episodes for evaluation (default: 10)",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=100_000,
        help="Frequency of evaluation during training (default: 50000)",
    )

    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Name of the Optuna study (default: auto-generated)",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Storage URL for Optuna study (e.g., 'sqlite:///optuna.db')",
    )
    parser.add_argument(
        "--load-if-exists",
        action="store_true",
        help="Load existing study if it exists",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Directory for logs and checkpoints (default: ./logs)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Weights & Biases project name (default: None, disabled)",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases entity/team name",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for training (default: auto)",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level (default: 1)",
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    if args.algo not in ALGORITHM_REGISTRY:
        raise ValueError(
            f"Algorithm '{args.algo}' not found. Available: {list(ALGORITHM_REGISTRY.keys())}"
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
    )

    algorithm_class, hyperparameter_sampler = ALGORITHM_REGISTRY[args.algo]

    optimizer = RLHyperParamOptimizer(
        algorithm=algorithm_class,
        env_name=args.env_name,
        hyperparameter_sampler=hyperparameter_sampler,
        n_timesteps=args.n_timesteps,
        n_eval_episodes=args.n_eval_episodes,
        eval_freq=args.eval_freq,
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=args.load_if_exists,
        verbose=args.verbose,
        seed=args.seed,
        device=args.device,
        log_dir=args.log_dir,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )

    optimizer.optimize(n_trials=args.n_trials)

    best_params_path = f"{args.log_dir}/best_params_{args.algo}.json"
    best_params = optimizer.get_best_params()
    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    logger.info(f"Best parameters saved to: {best_params_path}")
    logger.info(f"Best mean reward: {optimizer.get_best_value():.2f}")


if __name__ == "__main__":
    main()
