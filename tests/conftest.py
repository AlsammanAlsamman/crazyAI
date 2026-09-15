import pytest

from crazyai.toolkit.registry import build_toolkit


@pytest.fixture
def tk():
    return build_toolkit(seed=123)
