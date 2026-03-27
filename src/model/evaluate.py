"""
CLI script for model evaluation.

Usage:
    python -m model.evaluate --checkpoint checkpoints/bc_best.pth --test_data src/data/test_session
    python -m model.evaluate --checkpoint checkpoints/rl_best.pth --online --carla_host 172.28.224.1
"""

import argparse
import logging

import torch

from model.bc_model import BehavioralCloningModel
from model.checkpoint import CheckpointManager
from model.evaluator import ModelEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate driving model")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--test_data", help="Test session directory (offline eval)")
    parser.add_argument("--online", action="store_true", help="Run online eval in CARLA")
    parser.add_argument("--carla_host", default="localhost")
    parser.add_argument("--carla_port", type=int, default=2000)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load model
    model = BehavioralCloningModel(pretrained=False)
    ckpt_mgr = CheckpointManager()
    metadata = ckpt_mgr.load(args.checkpoint, model, device=args.device)
    logger.info("Loaded %s checkpoint (epoch %d)", metadata["model_type"], metadata["epoch"])

    evaluator = ModelEvaluator(device=args.device)

    if args.test_data:
        from model.dataset import DataLoaderFactory
        _, test_loader = DataLoaderFactory.create_dataloaders(
            args.test_data, batch_size=args.batch_size, num_workers=0, augment=False
        )
        metrics = evaluator.evaluate_offline(model, test_loader)
        print("\n=== Offline Evaluation ===")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    if args.online:
        from model.carla_gym_env import CARLAGymEnv
        env = CARLAGymEnv(host=args.carla_host, port=args.carla_port)
        metrics = evaluator.evaluate_online(model, env, num_episodes=args.num_episodes)
        print("\n=== Online Evaluation ===")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        env.close()


if __name__ == "__main__":
    main()
