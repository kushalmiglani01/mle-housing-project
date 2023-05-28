import os

import pytest


@pytest.fixture
def data_path():
    """Fixture to return the path to the test data directory."""
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "tests", "data"
    )
