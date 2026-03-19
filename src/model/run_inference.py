"""
CLI script for real-time CARLA inference.

Usage:
    python -m model.run_inference --checkpoint checkpoints/bc_best.pth
    python -m model.run_inference --checkpoint checkpoints/bc_best.pth --carla_host 172.28.224.1
"""

import argparse
import logging

import torch

from model.inference import BCInferenceEngine, CARLAControlLoop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run BC inference in CARLA")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--carla_host", default="localhost")
    parser.add_argument("--carla_port", type=int, default=2000)
    parser.add_argument("--max_steps", type=int, default=None, help="Max control steps (None=infinite)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    engine = BCInferenceEngine(args.checkpoint, device=args.device)
    logger.info("Loaded model (epoch %d)", engine.metadata["epoch"])

    loop = CARLAControlLoop(engine, host=args.carla_host, port=args.carla_port)

    try:
        loop.run(max_steps=args.max_steps)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        loop.stop()


if __name__ == "__main__":
    main()
