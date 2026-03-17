"""
Pytest configuration and shared fixtures for data pipeline tests.
"""

import pytest
from hypothesis import settings

# Configure Hypothesis for property-based testing
# Minimum 100 iterations per property test as specified in design
settings.register_profile("default", max_examples=100)
settings.load_profile("default")
