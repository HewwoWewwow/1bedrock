"""Pytest configuration and shared fixtures."""

import pytest
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.fixtures.test_inputs import (
    get_spec_test_inputs,
    get_spec_test_inputs_with_incentives,
)


@pytest.fixture
def spec_inputs():
    """Get spec test inputs without incentives."""
    return get_spec_test_inputs()


@pytest.fixture
def spec_inputs_with_incentives():
    """Get spec test inputs with Tier 2 incentives."""
    return get_spec_test_inputs_with_incentives()
