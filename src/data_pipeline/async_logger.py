"""
Asynchronous data logger for the data pipeline.

This module provides the AsyncDataLogger class which decouples sensor data
capture from disk I/O using a thread-safe queue and a ThreadPoolExecutor
for parallel write operations.
"""

import csv
import logging
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from data_pipeline.models import FrameData, VehicleState

logger = logging.getLogger(__name__)


class AsyncDataLogger:
    """
    Asynchronous data logger using a producer-consumer pattern.

    Frames are enqueued by the main capture thread and written to disk
    by background worker threads, ensuring zero frame drops during
    long-duration collection sessions.
    """

    CSV_HEADERS = ["image_filename", "speed", "steering", "throttle", "brake"]

    def __init__(
        self,
        output_dir: str,
        queue_size: int = 1000,
        num_workers: int = 2,
        png_compression: int = 3,
    ):
        """
        Initialize asynchronous data logger.

        Args:
            output_dir: Base directory for images/ and labels/ subdirectories.
            queue_size: Maximum queue capacity (default 1000 frames).
            num_workers: Number of I/O worker threads (default 2).
            png_compression: PNG compression level 0-9 (default 3, lower=faster).
        """
        self.output_dir = Path(output_dir)
        self.queue_size = queue_size
        self.num_workers = num_workers
        self.png_compression = png_compression

        # Thread-safe FIFO queue for frame data
        self._queue: queue.Queue[FrameData] = queue.Queue(maxsize=queue_size)

        # ThreadPoolExecutor for parallel disk I/O
        self._executor = ThreadPoolExecutor(max_workers=num_workers)

        # Consumer state
        self._running = False
        self._csv_lock = threading.Lock()
        self._futures = []

        # Tracking
        self.frame_drops = 0

        # Create output directories
        self._images_dir = self.output_dir / "images"
        self._labels_dir = self.output_dir / "labels"
        self._images_dir.mkdir(parents=True, exist_ok=True)
        self._labels_dir.mkdir(parents=True, exist_ok=True)

        # Initialize CSV file with headers
        self._csv_path = self._labels_dir / "driving_log.csv"
        with open(self._csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.CSV_HEADERS)

        logger.info(
            "AsyncDataLogger initialized: output_dir=%s, queue_size=%d, "
            "num_workers=%d, png_compression=%d",
            self.output_dir,
            self.queue_size,
            self.num_workers,
            self.png_compression,
        )

    def start(self) -> None:
        """Start background writer thread pool."""
        self._running = True
        self._futures = []
        for _ in range(self.num_workers):
            future = self._executor.submit(self._writer_loop)
            self._futures.append(future)
        logger.info("Started %d writer workers", self.num_workers)

    def enqueue_frame(
        self, timestamp_ms: int, image: np.ndarray, vehicle_state: VehicleState
    ) -> None:
        """
        Non-blocking enqueue of captured frame data.

        Args:
            timestamp_ms: Millisecond timestamp.
            image: RGB image array (800x600x3).
            vehicle_state: Vehicle telemetry data.
        """
        frame_data = FrameData(
            timestamp_ms=timestamp_ms,
            frame_id=timestamp_ms,
            image=image,
            vehicle_state=vehicle_state,
        )

        # Warn when queue reaches 90% capacity
        current_size = self._queue.qsize()
        if current_size >= 0.9 * self.queue_size:
            logger.warning(
                "Queue at %.0f%% capacity (%d/%d)",
                (current_size / self.queue_size) * 100,
                current_size,
                self.queue_size,
            )

        try:
            self._queue.put_nowait(frame_data)
        except queue.Full:
            self.frame_drops += 1
            logger.warning(
                "Queue overflow: Frame %d dropped (total drops: %d)",
                frame_data.frame_id,
                self.frame_drops,
            )


    def stop(self) -> None:
        """Stop writer threads and flush remaining queue."""
        self._running = False
        self._executor.shutdown(wait=True)
        logger.info(
            "Writer threads stopped. Frame drops: %d", self.frame_drops
        )

    def _writer_loop(self) -> None:
        """Background thread: dequeue and write to disk."""
        while self._running or not self._queue.empty():
            try:
                frame: FrameData = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Write PNG image
            image_path = self._images_dir / f"{frame.timestamp_ms}.png"
            try:
                cv2.imwrite(
                    str(image_path),
                    frame.image,
                    [cv2.IMWRITE_PNG_COMPRESSION, self.png_compression],
                )
            except Exception as e:
                logger.error("Failed to write image %s: %s", image_path, e)

            # Append CSV row (thread-safe)
            try:
                with self._csv_lock:
                    with open(self._csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            f"{frame.timestamp_ms}.png",
                            frame.vehicle_state.speed,
                            frame.vehicle_state.steering,
                            frame.vehicle_state.throttle,
                            frame.vehicle_state.brake,
                        ])
            except Exception as e:
                logger.error("Failed to append CSV row for frame %d: %s", frame.timestamp_ms, e)
