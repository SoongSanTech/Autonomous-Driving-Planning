"""
CLI script for RL (PPO) training with BC warm-start.

Usage:
    python -m model.train_rl --bc_checkpoint checkpoints/bc_best.pth
    python -m model.train_rl --bc_checkpoint checkpoints/bc_best.pth --episodes 3000 --carla_host 172.28.224.1
"""

import argparse
import logging

import torch

from model.carla_gym_env import CARLAGymEnv
from model.rl_policy import RLPolicyNetwork
from model.rl_trainer import RLTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train RL driving agent (PPO)")
    parser.add_argument("--bc_checkpoint", required=True, help="BC checkpoint path for warm-start")
    parser.add_argument("--carla_host", default="localhost")
    parser.add_argument("--carla_port", type=int, default=2000)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--frozen_episodes", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--finetune_lr", type=float, default=1e-5)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--checkpoint_interval", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    logger.info("Starting RL training: %s", args)

    # Load BC warm-start policy
    policy = RLPolicyNetwork.from_bc_checkpoint(args.bc_checkpoint, device=args.device)

    # Create CARLA environment
    env = CARLAGymEnv(host=args.carla_host, port=args.carla_port)

    # Train
    trainer = RLTrainer(
        policy, env,
        lr=args.lr,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )

    result = trainer.train(
        num_episodes=args.episodes,
        frozen_episodes=args.frozen_episodes,
        finetune_lr=args.finetune_lr,
        checkpoint_interval=args.checkpoint_interval,
    )

    logger.info("RL training complete. Best avg reward: %.2f", result["best_avg_reward"])
    logger.info("Best checkpoint: %s", result["best_checkpoint"])

    env.close()


if __name__ == "__main__":
    main()
