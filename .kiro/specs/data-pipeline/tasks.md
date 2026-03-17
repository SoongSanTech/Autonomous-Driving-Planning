# Implementation Plan: Data Pipeline

## Overview

This implementation plan breaks down the data-pipeline feature into discrete coding tasks. The system will be built incrementally, starting with core data structures, then the three main subsystems (Synchronous Mode Controller, Asynchronous Data Logger, Episode Manager), followed by the main orchestrator and integration. Each major component includes property-based tests using Hypothesis to validate correctness properties from the design document.

The implementation follows a Producer-Consumer architecture with thread-safe queue operations, ensuring zero frame drops during long-duration data collection sessions.

## Tasks

- [x] 1. Set up project structure and core data models
  - Create directory structure: `data_pipeline/` module with `__init__.py`
  - Create `data_pipeline/models.py` with `VehicleState` and `FrameData` dataclasses
  - Set up testing framework: `tests/` directory with `pytest` and `hypothesis` configuration
  - Create `requirements.txt` with dependencies: `carla`, `numpy`, `opencv-python`, `hypothesis`, `pytest`
  - _Requirements: 1.6, 1.3_

- [ ]* 1.1 Write property test for VehicleState completeness
  - **Property 5: Vehicle State Completeness**
  - **Validates: Requirements 1.6**
  - Generate random vehicle states, verify all required fields (speed, steering, throttle, brake) are present
  - _Requirements: 1.6_

- [ ]* 1.2 Write property test for timestamp precision
  - **Property 3: Timestamp Precision**
  - **Validates: Requirements 1.4**
  - Generate random timestamps, verify they are integer values representing milliseconds
  - _Requirements: 1.4_

- [ ] 2. Implement SynchronousModeController
  - [x] 2.1 Create `data_pipeline/sync_controller.py` with `SynchronousModeController` class
    - Implement `__init__` to accept CARLA world and tick_rate_hz parameter (default 10.0)
    - Implement `enable_synchronous_mode()` to configure world settings with fixed_delta_seconds
    - Implement `tick()` to advance simulation and return frame_id
    - Implement `get_timestamp_ms()` to return millisecond-precision simulation time
    - _Requirements: 2.1, 2.3, 1.4_

  - [ ]* 2.2 Write property test for timing consistency
    - **Property 7: Timing Consistency**
    - **Validates: Requirements 2.3**
    - Generate sequences of consecutive ticks, verify time differences are within 90-110ms range
    - _Requirements: 2.3_

  - [ ]* 2.3 Write unit tests for SynchronousModeController
    - Test tick rate configuration (10Hz default)
    - Test synchronous mode settings application
    - Test timestamp generation from simulation time
    - _Requirements: 2.1, 1.4_

- [ ] 3. Implement AsyncDataLogger with thread-safe queue
  - [x] 3.1 Create `data_pipeline/async_logger.py` with `AsyncDataLogger` class
    - Implement `__init__` to accept output_dir, queue_size, num_workers (default 2), and png_compression (default 3) parameters
    - Create thread-safe `queue.Queue` instance
    - Initialize ThreadPoolExecutor with num_workers for parallel I/O
    - Implement directory creation for `_out/images/` and `_out/labels/`
    - Initialize CSV file with headers: image_filename, speed, steering, throttle, brake
    - _Requirements: 3.2, 3.4, 4.2, 4.3_

  - [x] 3.2 Implement producer methods (enqueue operations)
    - Implement `enqueue_frame()` with non-blocking `put_nowait()` and overflow handling
    - Track frame drop counter for queue overflow events
    - Log warnings when queue reaches 90% capacity
    - _Requirements: 6.4_

  - [x] 3.3 Implement consumer methods (disk I/O operations)
    - Implement `start()` to launch background writer thread pool
    - Implement `_writer_loop()` to dequeue frames and write to disk
    - Implement PNG image writing using OpenCV with cv2.IMWRITE_PNG_COMPRESSION parameter set to 3
    - Implement CSV row appending with vehicle state data (thread-safe with lock)
    - Implement `stop()` to flush queue and terminate threads gracefully
    - _Requirements: 3.1, 3.3, 4.1, 4.4, 6.3_

  - [ ]* 3.4 Write property test for PNG format invariant
    - **Property 8: PNG Format Invariant**
    - **Validates: Requirements 3.1**
    - Generate random images, save to disk, verify successful PNG decode
    - _Requirements: 3.1_

  - [ ]* 3.5 Write property test for image file path correctness
    - **Property 9: Image File Path Correctness**
    - **Validates: Requirements 3.2, 3.3**
    - Generate random timestamps, verify saved files are at `_out/images/{timestamp}.png`
    - _Requirements: 3.2, 3.3_

  - [ ]* 3.6 Write property test for CSV persistence completeness
    - **Property 10: CSV Persistence Completeness**
    - **Validates: Requirements 4.1, 4.4**
    - Generate N random vehicle states, verify CSV contains exactly N rows (excluding header)
    - _Requirements: 4.1, 4.4_

  - [ ]* 3.7 Write property test for CSV schema invariant
    - **Property 11: CSV Schema Invariant**
    - **Validates: Requirements 4.3**
    - Generate random CSV files, verify exactly 5 columns with correct headers
    - _Requirements: 4.3_

  - [ ]* 3.8 Write unit tests for AsyncDataLogger
    - Test directory creation when paths don't exist
    - Test queue overflow handling and frame drop logging
    - Test graceful shutdown and queue flush
    - Test CSV file creation with correct headers
    - _Requirements: 3.4, 4.2, 6.4_

- [x] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 5. Implement EpisodeManager for scenario variation
  - [x] 5.1 Create `data_pipeline/episode_manager.py` with enums and `EpisodeManager` class
    - Define `WeatherPreset` enum with CLEAR, RAIN, FOG values
    - Define `TimeOfDay` enum with DAYTIME, NIGHT, BACKLIGHT values
    - Implement `__init__` to accept CARLA world and episode_duration_sec (default 300.0)
    - _Requirements: 5.2, 5.4_

  - [x] 5.2 Implement scenario configuration methods
    - Implement `apply_weather()` to set CARLA weather parameters with error handling
    - Implement `apply_time_of_day()` to set sun angle with error handling
    - Implement `start_new_episode()` to randomly select and apply weather and time of day
    - Implement `should_reset_episode()` to check if episode duration has elapsed
    - _Requirements: 5.1, 5.3, 5.5, 5.6_

  - [ ]* 5.3 Write property test for weather diversity
    - **Property 13: Weather Diversity**
    - **Validates: Requirements 5.2**
    - Generate multi-episode sessions (duration > 5 minutes), verify at least 2 different weather conditions applied
    - _Requirements: 5.2_

  - [ ]* 5.4 Write property test for time of day diversity
    - **Property 14: Time of Day Diversity**
    - **Validates: Requirements 5.4**
    - Generate multi-episode sessions (duration > 5 minutes), verify at least 2 different time of day settings applied
    - _Requirements: 5.4_

  - [ ]* 5.5 Write unit tests for EpisodeManager
    - Test episode reset timing (5-minute intervals)
    - Test weather randomization across episodes
    - Test time of day randomization across episodes
    - Test error handling for failed weather/time application
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [ ] 6. Implement DataPipeline main orchestrator
  - [x] 6.1 Create `data_pipeline/pipeline.py` with `DataPipeline` class
    - Implement `__init__` to accept carla_host, carla_port, output_dir, headless parameters
    - Initialize all subsystems: SynchronousModeController, AsyncDataLogger, EpisodeManager
    - _Requirements: 7.2, 8.1_

  - [x] 6.2 Implement connection and setup methods
    - Implement `connect()` with retry logic (exponential backoff, max 5 attempts)
    - Add WSL2 network handling: accept explicit host IP to avoid localhost mapping issues
    - Implement connection error handling with descriptive error messages
    - Implement `setup_sensors()` to attach camera sensor with callback
    - Configure camera resolution to 800x600 pixels
    - _Requirements: 7.1, 7.2, 7.3, 1.5_

  - [x] 6.3 Implement main collection loop
    - Implement `run()` method with duration parameter (default 3600 seconds)
    - Main loop: tick synchronous controller, capture frame data, enqueue to logger
    - Implement CARLA server crash detection via tick timeout
    - On crash detection: flush queue, save partial data, log frames saved count
    - Implement episode reset logic based on elapsed time
    - Track collection statistics: frames captured, frame drops, elapsed time
    - Implement headless mode logging to console
    - _Requirements: 2.1, 6.1, 6.2, 8.1, 8.2, 9.1, 9.2, 9.3, 9.4_

  - [x] 6.4 Implement shutdown and cleanup
    - Implement `shutdown()` with graceful termination
    - Register signal handlers for SIGINT and SIGTERM
    - Flush remaining queue items before exit (data preservation)
    - Close CSV file properly and disconnect from CARLA
    - Report collection statistics at shutdown (frames saved, drops, duration)
    - Implement file existence check to prevent overwriting existing data on restart
    - _Requirements: 7.4, 9.2, 9.3, 9.5, 9.6_

  - [ ]* 6.5 Write property test for complete frame capture
    - **Property 1: Complete Frame Capture**
    - **Validates: Requirements 1.1, 1.2**
    - Generate random collection cycles, verify each frame contains both image and vehicle state
    - _Requirements: 1.1, 1.2_

  - [ ]* 6.6 Write property test for timestamp synchronization
    - **Property 2: Timestamp Synchronization**
    - **Validates: Requirements 1.3**
    - Generate random frames, verify image and vehicle state have identical timestamps
    - _Requirements: 1.3_

  - [ ]* 6.7 Write property test for image resolution invariant
    - **Property 4: Image Resolution Invariant**
    - **Validates: Requirements 1.5**
    - Generate random captured images, verify dimensions are exactly 800x600 pixels
    - _Requirements: 1.5_

  - [ ]* 6.8 Write property test for collection frequency
    - **Property 6: Collection Frequency**
    - **Validates: Requirements 2.1, 2.2, 6.2**
    - Generate random time intervals, verify system completes at least (T × 10) cycles
    - _Requirements: 2.1, 2.2, 6.2_

  - [ ]* 6.9 Write unit tests for DataPipeline
    - Test connection error handling when CARLA is unreachable
    - Test WSL2 network handling with explicit host IP
    - Test sensor setup with correct camera resolution
    - Test headless mode execution without display
    - Test graceful shutdown on interrupt signal
    - Test CARLA server crash detection and data preservation
    - Test no data overwrite on restart after crash
    - _Requirements: 7.3, 1.5, 8.1, 9.1, 9.5, 9.6_

- [x] 7. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 8. Create integration tests and end-to-end validation
  - [x] 8.1 Write integration test for 10-second collection session
    - Run complete pipeline for 10 seconds
    - Verify image files created in `_out/images/`
    - Verify CSV file created in `_out/labels/driving_log.csv`
    - Verify file count matches expected frame count (approximately 100 frames)
    - _Requirements: 2.1, 3.1, 3.2, 3.3, 4.1, 4.2_

  - [x] 8.2 Write integration test for episode transitions
    - Run pipeline for 6 minutes (2 episodes)
    - Verify seamless weather and time of day changes
    - Verify no frame drops during episode reset
    - _Requirements: 5.5, 5.6, 6.4_

  - [ ]* 8.3 Write property test for image-label correspondence
    - **Property 12: Image-Label Correspondence**
    - **Validates: Requirements 4.4**
    - Generate random datasets, verify each CSV row references an existing PNG file
    - _Requirements: 4.4_

  - [ ]* 8.4 Write property test for zero frame drop
    - **Property 15: Zero Frame Drop**
    - **Validates: Requirements 6.3, 6.4**
    - Generate random session durations, verify saved frame count equals expected frame count
    - _Requirements: 6.3, 6.4_

  - [ ]* 8.5 Write property test for connection persistence
    - **Property 16: Connection Persistence**
    - **Validates: Requirements 7.4**
    - Generate random collection sessions, verify TCP connection remains active throughout
    - _Requirements: 7.4_

  - [ ]* 8.6 Write property test for headless logging
    - **Property 17: Headless Logging**
    - **Validates: Requirements 8.2**
    - Generate random headless sessions, verify log output is produced
    - _Requirements: 8.2_

  - [ ]* 8.7 Write property test for fault tolerance - data preservation
    - **Property 18: Fault Tolerance - Data Preservation**
    - **Validates: Requirements 9.2, 9.3, 9.5**
    - Simulate CARLA server crashes at random times, verify all frames before crash are saved
    - _Requirements: 9.2, 9.3, 9.5_

  - [ ]* 8.8 Write property test for fault tolerance - no data overwrite
    - **Property 19: Fault Tolerance - No Data Overwrite**
    - **Validates: Requirements 9.6**
    - Generate random restart scenarios, verify existing files are not overwritten
    - _Requirements: 9.6_

- [ ] 9. Create CLI entry point and documentation
  - [x] 9.1 Create `data_pipeline/cli.py` with command-line interface
    - Use `argparse` to accept parameters: host, port, output_dir, duration, headless
    - Implement main() function to instantiate and run DataPipeline
    - Add `if __name__ == "__main__"` block for direct execution
    - _Requirements: 7.2, 8.1_

  - [x] 9.2 Create README.md with usage instructions
    - Document system requirements (Windows + WSL2, CARLA 0.9.15, Python 3.8+)
    - Document installation steps (pip install requirements)
    - Document CLI usage examples with explicit host IP for WSL2
    - Document output directory structure
    - Document troubleshooting common issues:
      * WSL2 localhost mapping issues (use Windows host IP)
      * Connection errors and retry logic
      * Queue overflow and PNG compression tuning
      * CARLA server crash recovery and data preservation
    - _Requirements: 7.1, 7.2, 8.1, 9.5_

- [x] 10. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 11. Extend pipeline to multi-camera configuration (5 cameras)
  - [ ] 11.1 Update `data_pipeline/models.py` with multi-camera FrameData
    - Add `MultiCameraFrameData` dataclass with front_image, avm_front_image, avm_rear_image, avm_left_image, avm_right_image fields
    - Front image shape: (600, 800, 3), AVM image shapes: (300, 400, 3)
    - Maintain backward compatibility with existing `FrameData` for single-camera tests
    - _Requirements: 1.1, 1.2, 1.6, 1.7_

  - [ ] 11.2 Update `data_pipeline/pipeline.py` setup_sensors() for 5 cameras
    - Attach Front RGB camera at (x=1.5, z=2.4), FOV 90°, 800×600
    - Attach AVM Front camera at (x=2.0, z=0.5), pitch=-90°, FOV 120°, 400×300
    - Attach AVM Rear camera at (x=-2.0, z=0.5), pitch=-90°, yaw=180°, FOV 120°, 400×300
    - Attach AVM Left camera at (y=-1.0, z=0.5), pitch=-90°, yaw=-90°, FOV 120°, 400×300
    - Attach AVM Right camera at (y=1.0, z=0.5), pitch=-90°, yaw=90°, FOV 120°, 400×300
    - Register callbacks for all 5 cameras with thread-safe image storage
    - _Requirements: 10.1, 10.2, 10.3_

  - [ ] 11.3 Update `data_pipeline/async_logger.py` for multi-camera I/O
    - Create subdirectories: front/, avm_front/, avm_rear/, avm_left/, avm_right/, bev/, labels/
    - Update enqueue to accept multi-camera frame data
    - Update writer loop to save 5 images per frame (front + 4 AVM)
    - Adjust queue size and worker count for increased I/O throughput
    - _Requirements: 3.2, 3.3, 10.4_

  - [ ] 11.4 Update `data_pipeline/pipeline.py` run() loop for multi-camera capture
    - Wait for all 5 camera images per tick (synchronous mode guarantees delivery)
    - Log warning if any AVM camera fails to deliver within timeout
    - Enqueue complete multi-camera frame data
    - _Requirements: 10.3, 10.5_

  - [ ] 11.5 Update existing tests for multi-camera compatibility
    - Update test fixtures and mocks to support 5-camera configuration
    - Add tests for AVM camera attachment and image capture
    - Add tests for multi-camera directory structure creation
    - Verify all 89 existing tests still pass (backward compatibility)
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

- [ ] 12. Final checkpoint - Ensure all tests pass after multi-camera extension
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Property-based tests use Hypothesis framework with minimum 100 iterations
- Integration tests require CARLA server running on Windows Host
- The system is designed for Python 3.8+ with type hints
- All property tests include explicit property number and requirements validation annotations
