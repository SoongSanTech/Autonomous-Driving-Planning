"""
Unit tests for core data models.
"""

import pytest
import numpy as np
from data_pipeline.models import VehicleState, FrameData


@pytest.mark.unit
class TestVehicleState:
    """Tests for VehicleState dataclass."""
    
    def test_vehicle_state_creation(self):
        """Verify VehicleState can be created with all required fields."""
        state = VehicleState(
            speed=15.5,
            steering=0.12,
            throttle=0.5,
            brake=0.0
        )
        
        assert state.speed == 15.5
        assert state.steering == 0.12
        assert state.throttle == 0.5
        assert state.brake == 0.0
    
    def test_vehicle_state_completeness(self):
        """Verify VehicleState contains all required fields (Property 5)."""
        state = VehicleState(speed=10.0, steering=0.0, throttle=0.3, brake=0.0)
        
        # Validates: Requirements 1.6
        assert hasattr(state, 'speed')
        assert hasattr(state, 'steering')
        assert hasattr(state, 'throttle')
        assert hasattr(state, 'brake')


@pytest.mark.unit
class TestFrameData:
    """Tests for FrameData dataclass."""
    
    def test_frame_data_creation(self):
        """Verify FrameData can be created with all required fields."""
        vehicle_state = VehicleState(
            speed=15.5,
            steering=0.12,
            throttle=0.5,
            brake=0.0
        )
        
        image = np.zeros((600, 800, 3), dtype=np.uint8)
        
        frame = FrameData(
            timestamp_ms=1234567890,
            frame_id=42,
            image=image,
            vehicle_state=vehicle_state
        )
        
        assert frame.timestamp_ms == 1234567890
        assert frame.frame_id == 42
        assert frame.image.shape == (600, 800, 3)
        assert frame.vehicle_state.speed == 15.5
    
    def test_frame_data_image_shape(self):
        """Verify FrameData image has correct dimensions (Property 4)."""
        vehicle_state = VehicleState(speed=10.0, steering=0.0, throttle=0.3, brake=0.0)
        image = np.zeros((600, 800, 3), dtype=np.uint8)
        
        frame = FrameData(
            timestamp_ms=1000000,
            frame_id=1,
            image=image,
            vehicle_state=vehicle_state
        )
        
        # Validates: Requirements 1.5
        assert frame.image.shape == (600, 800, 3), \
            f"Expected (600, 800, 3), got {frame.image.shape}"
        assert frame.image.dtype == np.uint8
    
    def test_timestamp_precision(self):
        """Verify timestamp is an integer value in milliseconds (Property 3)."""
        vehicle_state = VehicleState(speed=10.0, steering=0.0, throttle=0.3, brake=0.0)
        image = np.zeros((600, 800, 3), dtype=np.uint8)
        
        frame = FrameData(
            timestamp_ms=1234567890,
            frame_id=1,
            image=image,
            vehicle_state=vehicle_state
        )
        
        # Validates: Requirements 1.4
        assert isinstance(frame.timestamp_ms, int)
        assert frame.timestamp_ms > 0
