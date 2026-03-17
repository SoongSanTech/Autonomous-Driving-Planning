"""Command-line interface for the CARLA Data Collection Pipeline."""

import argparse
import logging

from data_pipeline.pipeline import DataPipeline


def main():
    """Parse CLI arguments and run the data collection pipeline."""
    parser = argparse.ArgumentParser(
        description="CARLA Data Collection Pipeline"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="CARLA server host (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=2000,
        help="CARLA server port (default: 2000)",
    )
    parser.add_argument(
        "--output-dir",
        default="src/data",
        help="Output directory (default: src/data)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3600.0,
        help="Collection duration in seconds (default: 3600)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run without display (default: True)",
    )
    parser.add_argument(
        "--no-headless",
        action="store_false",
        dest="headless",
        help="Run with display",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    pipeline = DataPipeline(
        carla_host=args.host,
        carla_port=args.port,
        output_dir=args.output_dir,
        headless=args.headless,
    )

    try:
        pipeline.connect()
        pipeline.setup_sensors()
        pipeline.run(duration_sec=args.duration)
    finally:
        pipeline.shutdown()


if __name__ == "__main__":
    main()
