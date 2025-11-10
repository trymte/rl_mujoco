"""Main entry point for RL training pipeline."""

import argparse
import logging

from rl.ppo.config import TrainingConfig
from rl.ppo.training import TrainingPipeline

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="RL Training Pipeline for Humanoid Locomotion"
    )

    parser.add_argument(
        "--env-name",
        type=str,
        default="Humanoid-v5",
        help="Gymnasium environment name (default: Humanoid-v5)",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Directory for logs and checkpoints (default: ./logs)",
    )
    parser.add_argument(
        "--tensorboard-log",
        type=str,
        default="./tensorboard",
        help="Directory for tensorboard logs (default: ./tensorboard)",
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
        "--eval-only",
        action="store_true",
        help="Only evaluate a trained model (requires --checkpoint-path)",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to model checkpoint for resuming or evaluation",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of episodes for evaluation (default: 10)",
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

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
    )

    config = TrainingConfig(
        env_name=args.env_name,
        log_dir=args.log_dir,
        tensorboard_log=args.tensorboard_log,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        checkpoint_path=args.checkpoint_path,
        seed=args.seed,
        device=args.device,
        eval_episodes=args.eval_episodes,
    )

    pipeline = TrainingPipeline(config)

    if args.eval_only:
        if args.checkpoint_path is None:
            logger.info("Error: --checkpoint-path is required for evaluation mode")
            return

        logger.info("=" * 60)
        logger.info("Evaluation Mode")
        logger.info("=" * 60)
        pipeline.setup_environments()
        pipeline.setup_model()
        pipeline.evaluate(n_episodes=args.eval_episodes, save_gif=True)
    else:
        logger.info("=" * 60)
        logger.info("RL Training Pipeline for Humanoid Locomotion")
        logger.info("=" * 60)
        logger.info(f"Config: {config}")
        logger.info(f"Environment: {config.env_name}")
        logger.info(f"Log directory: {config.log_dir}")
        logger.info("=" * 60)

        pipeline.setup_environments()
        pipeline.setup_model()
        pipeline.train()

        logger.info("Evaluating final model...")
        pipeline.evaluate(n_episodes=args.eval_episodes, save_gif=True)


if __name__ == "__main__":
    main()
