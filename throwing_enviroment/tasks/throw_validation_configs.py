"""Validation test configurations for the throwing primitive.

10 predefined deterministic target positions covering the workspace:
  x ∈ [0.0, 0.5], y ∈ [1.0, 1.6]

Each test specifies a fixed basket position for reproducible evaluation.
"""

from dataclasses import dataclass
from typing import List, Optional

SUCCESS_THRESHOLD_DEFAULT = 0.15


@dataclass
class ThrowTestConfig:
    test_id: int
    name: str
    target_x: float
    target_y: float


THROW_TESTS: List[ThrowTestConfig] = [
    ThrowTestConfig(test_id=1, name="Center near", target_x=0.00, target_y=1.10),
    ThrowTestConfig(test_id=2, name="Center far", target_x=0.00, target_y=1.50),
    ThrowTestConfig(test_id=3, name="Right near", target_x=0.40, target_y=1.10),
    ThrowTestConfig(test_id=4, name="Right far", target_x=0.40, target_y=1.50),
    ThrowTestConfig(test_id=5, name="Center-right mid", target_x=0.20, target_y=1.30),
    ThrowTestConfig(test_id=6, name="Slight-right near", target_x=0.10, target_y=1.10),
    ThrowTestConfig(test_id=7, name="Right mid-far", target_x=0.30, target_y=1.40),
    ThrowTestConfig(test_id=8, name="Far-right near", target_x=0.45, target_y=1.20),
    ThrowTestConfig(test_id=9, name="Center far edge", target_x=0.05, target_y=1.55),
    ThrowTestConfig(test_id=10, name="Center-right closest", target_x=0.25, target_y=1.00),
]


def get_test_count() -> int:
    return len(THROW_TESTS)


def get_test_config(index: int) -> Optional[ThrowTestConfig]:
    if 1 <= index <= len(THROW_TESTS):
        return THROW_TESTS[index - 1]
    return None
