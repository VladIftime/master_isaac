"""
Validation test configuration definitions for push-task evaluation.

Provides pre-defined deterministic test configurations for:
  1. Single-object push (10 tests)
  2. Multi-object push with 2-4 extra objects (30 tests)
  3. T-push (Diffusion Policy-style) (10 tests)

Total: 50 test scenes + index 0 = free-play mode.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ObjectStart:
    """Start configuration for a single pushed object."""

    x: float
    y: float
    yaw_deg: float = 0.0  # degrees, rotation around Z


@dataclass
class PushTestConfig:
    """Configuration for a single validation test scene."""

    name: str  # e.g. "Single Push", "Multi-Object (2 extra)", "T-Push"
    category: str  # "single", "multi_2", "multi_3", "multi_4", "tpush"
    test_id: int  # 1-10 within its category

    # Main object (the one to push)
    block_type: Optional[str] = None  # None = random from BLOCK_FILES
    main_start: ObjectStart = field(default_factory=lambda: ObjectStart(0.0, 0.5))
    main_goal_x: float = 0.0
    main_goal_y: float = 0.5
    main_goal_yaw_deg: float = 0.0

    # Extra objects (distractors)
    extra_starts: List[ObjectStart] = field(default_factory=list)

    # End-zone position (where the gripper must go to complete the test)
    end_zone_x: float = 0.05
    end_zone_y: float = 0.30

    # Whether this test uses the T-shaped block
    is_t_block: bool = False


# Workspace bounds (from test_curobo_follow_target.py)
_WS_X_MIN, _WS_X_MAX = -0.65, 0.65
_WS_Y_MIN, _WS_Y_MAX = 0.20, 0.75

# Helper: clamp a value within a range with optional margin from edges
def _clamp(v, lo, hi, margin=0.05):
    return max(lo + margin, min(hi - margin, v))


def _validate_pos(x, y, margin=0.08):
    """Validate and clamp a position to workspace bounds."""
    return _clamp(x, _WS_X_MIN, _WS_X_MAX, margin), _clamp(y, _WS_Y_MIN, _WS_Y_MAX, margin)


# =========================================================================
# 1. Single Object Push Tests (indices 1-10)
# =========================================================================
SINGLE_PUSH_TESTS = [
    PushTestConfig(
        name="Single Push",
        category="single",
        test_id=1,
        main_start=ObjectStart(-0.30, 0.70),
        main_goal_x=0.30, main_goal_y=0.60, main_goal_yaw_deg=0,
    ),
    PushTestConfig(
        name="Single Push",
        category="single",
        test_id=2,
        main_start=ObjectStart(0.30, 0.60),
        main_goal_x=-0.30, main_goal_y=0.70, main_goal_yaw_deg=180,
    ),
    PushTestConfig(
        name="Single Push",
        category="single",
        test_id=3,
        main_start=ObjectStart(-0.20, 0.50),
        main_goal_x=0.20, main_goal_y=0.75, main_goal_yaw_deg=90,
    ),
    PushTestConfig(
        name="Single Push",
        category="single",
        test_id=4,
        main_start=ObjectStart(0.20, 0.75),
        main_goal_x=-0.20, main_goal_y=0.50, main_goal_yaw_deg=-90,
    ),
    PushTestConfig(
        name="Single Push",
        category="single",
        test_id=5,
        main_start=ObjectStart(-0.35, 0.65),
        main_goal_x=0.35, main_goal_y=0.55, main_goal_yaw_deg=45,
    ),
    PushTestConfig(
        name="Single Push",
        category="single",
        test_id=6,
        main_start=ObjectStart(0.35, 0.55),
        main_goal_x=-0.35, main_goal_y=0.65, main_goal_yaw_deg=-45,
    ),
    PushTestConfig(
        name="Single Push",
        category="single",
        test_id=7,
        main_start=ObjectStart(-0.10, 0.55),
        main_goal_x=0.25, main_goal_y=0.70, main_goal_yaw_deg=0,
    ),
    PushTestConfig(
        name="Single Push",
        category="single",
        test_id=8,
        main_start=ObjectStart(0.25, 0.70),
        main_goal_x=-0.10, main_goal_y=0.55, main_goal_yaw_deg=0,
    ),
    PushTestConfig(
        name="Single Push",
        category="single",
        test_id=9,
        main_start=ObjectStart(0.00, 0.50),
        main_goal_x=0.00, main_goal_y=0.70, main_goal_yaw_deg=0,
    ),
    PushTestConfig(
        name="Single Push",
        category="single",
        test_id=10,
        main_start=ObjectStart(0.00, 0.70),
        main_goal_x=0.00, main_goal_y=0.50, main_goal_yaw_deg=90,
    ),
]

# =========================================================================
# 2. Multi-Object Push Tests (indices 11-40)
#    11-20: 2 extra objects  (3 total)
#    21-30: 3 extra objects  (4 total)
#    31-40: 4 extra objects  (5 total)
# =========================================================================

MULTI_2_TESTS = [
    PushTestConfig(
        name="Multi-Object (2 extra)",
        category="multi_2", test_id=1,
        main_start=ObjectStart(-0.20, 0.70),
        main_goal_x=0.25, main_goal_y=0.60, main_goal_yaw_deg=0,
        extra_starts=[ObjectStart(-0.30, 0.55), ObjectStart(0.10, 0.72)],
    ),
    PushTestConfig(
        name="Multi-Object (2 extra)",
        category="multi_2", test_id=2,
        main_start=ObjectStart(0.20, 0.55),
        main_goal_x=-0.25, main_goal_y=0.70, main_goal_yaw_deg=180,
        extra_starts=[ObjectStart(0.30, 0.72), ObjectStart(-0.10, 0.50)],
    ),
    PushTestConfig(
        name="Multi-Object (2 extra)",
        category="multi_2", test_id=3,
        main_start=ObjectStart(-0.30, 0.60),
        main_goal_x=0.20, main_goal_y=0.50, main_goal_yaw_deg=90,
        extra_starts=[ObjectStart(0.25, 0.70), ObjectStart(-0.10, 0.75)],
    ),
    PushTestConfig(
        name="Multi-Object (2 extra)",
        category="multi_2", test_id=4,
        main_start=ObjectStart(0.25, 0.70),
        main_goal_x=-0.30, main_goal_y=0.60, main_goal_yaw_deg=-90,
        extra_starts=[ObjectStart(-0.20, 0.50), ObjectStart(0.35, 0.55)],
    ),
    PushTestConfig(
        name="Multi-Object (2 extra)",
        category="multi_2", test_id=5,
        main_start=ObjectStart(0.00, 0.50),
        main_goal_x=0.30, main_goal_y=0.70, main_goal_yaw_deg=45,
        extra_starts=[ObjectStart(-0.35, 0.65), ObjectStart(0.10, 0.55)],
    ),
    PushTestConfig(
        name="Multi-Object (2 extra)",
        category="multi_2", test_id=6,
        main_start=ObjectStart(-0.15, 0.55),
        main_goal_x=0.35, main_goal_y=0.65, main_goal_yaw_deg=-45,
        extra_starts=[ObjectStart(0.15, 0.75), ObjectStart(-0.35, 0.70)],
    ),
    PushTestConfig(
        name="Multi-Object (2 extra)",
        category="multi_2", test_id=7,
        main_start=ObjectStart(-0.35, 0.70),
        main_goal_x=0.10, main_goal_y=0.55, main_goal_yaw_deg=0,
        extra_starts=[ObjectStart(0.30, 0.60), ObjectStart(-0.10, 0.70)],
    ),
    PushTestConfig(
        name="Multi-Object (2 extra)",
        category="multi_2", test_id=8,
        main_start=ObjectStart(0.35, 0.50),
        main_goal_x=-0.10, main_goal_y=0.70, main_goal_yaw_deg=0,
        extra_starts=[ObjectStart(-0.30, 0.55), ObjectStart(0.20, 0.72)],
    ),
    PushTestConfig(
        name="Multi-Object (2 extra)",
        category="multi_2", test_id=9,
        main_start=ObjectStart(-0.10, 0.72),
        main_goal_x=0.20, main_goal_y=0.55, main_goal_yaw_deg=0,
        extra_starts=[ObjectStart(0.30, 0.70), ObjectStart(-0.25, 0.50)],
    ),
    PushTestConfig(
        name="Multi-Object (2 extra)",
        category="multi_2", test_id=10,
        main_start=ObjectStart(0.10, 0.75),
        main_goal_x=-0.20, main_goal_y=0.50, main_goal_yaw_deg=90,
        extra_starts=[ObjectStart(-0.30, 0.60), ObjectStart(0.25, 0.55)],
    ),
]

MULTI_3_TESTS = [
    PushTestConfig(
        name="Multi-Object (3 extra)",
        category="multi_3", test_id=1,
        main_start=ObjectStart(-0.20, 0.70),
        main_goal_x=0.25, main_goal_y=0.60, main_goal_yaw_deg=0,
        extra_starts=[ObjectStart(-0.30, 0.55), ObjectStart(0.10, 0.72), ObjectStart(0.30, 0.50)],
    ),
    PushTestConfig(
        name="Multi-Object (3 extra)",
        category="multi_3", test_id=2,
        main_start=ObjectStart(0.20, 0.55),
        main_goal_x=-0.25, main_goal_y=0.70, main_goal_yaw_deg=180,
        extra_starts=[ObjectStart(0.30, 0.72), ObjectStart(-0.10, 0.50), ObjectStart(-0.35, 0.60)],
    ),
    PushTestConfig(
        name="Multi-Object (3 extra)",
        category="multi_3", test_id=3,
        main_start=ObjectStart(-0.30, 0.60),
        main_goal_x=0.20, main_goal_y=0.50, main_goal_yaw_deg=90,
        extra_starts=[ObjectStart(0.25, 0.70), ObjectStart(-0.10, 0.75), ObjectStart(0.35, 0.65)],
    ),
    PushTestConfig(
        name="Multi-Object (3 extra)",
        category="multi_3", test_id=4,
        main_start=ObjectStart(0.25, 0.70),
        main_goal_x=-0.30, main_goal_y=0.60, main_goal_yaw_deg=-90,
        extra_starts=[ObjectStart(-0.20, 0.50), ObjectStart(0.35, 0.55), ObjectStart(-0.35, 0.72)],
    ),
    PushTestConfig(
        name="Multi-Object (3 extra)",
        category="multi_3", test_id=5,
        main_start=ObjectStart(0.00, 0.50),
        main_goal_x=0.30, main_goal_y=0.70, main_goal_yaw_deg=45,
        extra_starts=[ObjectStart(-0.35, 0.65), ObjectStart(0.10, 0.55), ObjectStart(-0.20, 0.75)],
    ),
    PushTestConfig(
        name="Multi-Object (3 extra)",
        category="multi_3", test_id=6,
        main_start=ObjectStart(-0.15, 0.55),
        main_goal_x=0.35, main_goal_y=0.65, main_goal_yaw_deg=-45,
        extra_starts=[ObjectStart(0.15, 0.75), ObjectStart(-0.35, 0.70), ObjectStart(0.30, 0.55)],
    ),
    PushTestConfig(
        name="Multi-Object (3 extra)",
        category="multi_3", test_id=7,
        main_start=ObjectStart(-0.35, 0.70),
        main_goal_x=0.10, main_goal_y=0.55, main_goal_yaw_deg=0,
        extra_starts=[ObjectStart(0.30, 0.60), ObjectStart(-0.10, 0.70), ObjectStart(0.20, 0.50)],
    ),
    PushTestConfig(
        name="Multi-Object (3 extra)",
        category="multi_3", test_id=8,
        main_start=ObjectStart(0.35, 0.50),
        main_goal_x=-0.10, main_goal_y=0.70, main_goal_yaw_deg=0,
        extra_starts=[ObjectStart(-0.30, 0.55), ObjectStart(0.20, 0.72), ObjectStart(-0.20, 0.65)],
    ),
    PushTestConfig(
        name="Multi-Object (3 extra)",
        category="multi_3", test_id=9,
        main_start=ObjectStart(-0.10, 0.72),
        main_goal_x=0.20, main_goal_y=0.55, main_goal_yaw_deg=0,
        extra_starts=[ObjectStart(0.30, 0.70), ObjectStart(-0.25, 0.50), ObjectStart(0.35, 0.72)],
    ),
    PushTestConfig(
        name="Multi-Object (3 extra)",
        category="multi_3", test_id=10,
        main_start=ObjectStart(0.10, 0.75),
        main_goal_x=-0.20, main_goal_y=0.50, main_goal_yaw_deg=90,
        extra_starts=[ObjectStart(-0.30, 0.60), ObjectStart(0.25, 0.55), ObjectStart(-0.35, 0.50)],
    ),
]

MULTI_4_TESTS = [
    PushTestConfig(
        name="Multi-Object (4 extra)",
        category="multi_4", test_id=1,
        main_start=ObjectStart(-0.20, 0.70),
        main_goal_x=0.25, main_goal_y=0.60, main_goal_yaw_deg=0,
        extra_starts=[ObjectStart(-0.30, 0.55), ObjectStart(0.10, 0.72), ObjectStart(0.30, 0.50), ObjectStart(-0.35, 0.65)],
    ),
    PushTestConfig(
        name="Multi-Object (4 extra)",
        category="multi_4", test_id=2,
        main_start=ObjectStart(0.20, 0.55),
        main_goal_x=-0.25, main_goal_y=0.70, main_goal_yaw_deg=180,
        extra_starts=[ObjectStart(0.30, 0.72), ObjectStart(-0.10, 0.50), ObjectStart(-0.35, 0.60), ObjectStart(0.35, 0.65)],
    ),
    PushTestConfig(
        name="Multi-Object (4 extra)",
        category="multi_4", test_id=3,
        main_start=ObjectStart(-0.30, 0.60),
        main_goal_x=0.20, main_goal_y=0.50, main_goal_yaw_deg=90,
        extra_starts=[ObjectStart(0.25, 0.70), ObjectStart(-0.10, 0.75), ObjectStart(0.35, 0.65), ObjectStart(-0.35, 0.50)],
    ),
    PushTestConfig(
        name="Multi-Object (4 extra)",
        category="multi_4", test_id=4,
        main_start=ObjectStart(0.25, 0.70),
        main_goal_x=-0.30, main_goal_y=0.60, main_goal_yaw_deg=-90,
        extra_starts=[ObjectStart(-0.20, 0.50), ObjectStart(0.35, 0.55), ObjectStart(-0.35, 0.72), ObjectStart(0.30, 0.70)],
    ),
    PushTestConfig(
        name="Multi-Object (4 extra)",
        category="multi_4", test_id=5,
        main_start=ObjectStart(0.00, 0.50),
        main_goal_x=0.30, main_goal_y=0.70, main_goal_yaw_deg=45,
        extra_starts=[ObjectStart(-0.35, 0.65), ObjectStart(0.10, 0.55), ObjectStart(-0.20, 0.75), ObjectStart(0.35, 0.55)],
    ),
    PushTestConfig(
        name="Multi-Object (4 extra)",
        category="multi_4", test_id=6,
        main_start=ObjectStart(-0.15, 0.55),
        main_goal_x=0.35, main_goal_y=0.65, main_goal_yaw_deg=-45,
        extra_starts=[ObjectStart(0.15, 0.75), ObjectStart(-0.35, 0.70), ObjectStart(0.30, 0.55), ObjectStart(-0.25, 0.60)],
    ),
    PushTestConfig(
        name="Multi-Object (4 extra)",
        category="multi_4", test_id=7,
        main_start=ObjectStart(-0.35, 0.70),
        main_goal_x=0.10, main_goal_y=0.55, main_goal_yaw_deg=0,
        extra_starts=[ObjectStart(0.30, 0.60), ObjectStart(-0.10, 0.70), ObjectStart(0.20, 0.50), ObjectStart(0.35, 0.72)],
    ),
    PushTestConfig(
        name="Multi-Object (4 extra)",
        category="multi_4", test_id=8,
        main_start=ObjectStart(0.35, 0.50),
        main_goal_x=-0.10, main_goal_y=0.70, main_goal_yaw_deg=0,
        extra_starts=[ObjectStart(-0.30, 0.55), ObjectStart(0.20, 0.72), ObjectStart(-0.20, 0.65), ObjectStart(-0.35, 0.70)],
    ),
    PushTestConfig(
        name="Multi-Object (4 extra)",
        category="multi_4", test_id=9,
        main_start=ObjectStart(-0.10, 0.72),
        main_goal_x=0.20, main_goal_y=0.55, main_goal_yaw_deg=0,
        extra_starts=[ObjectStart(0.30, 0.70), ObjectStart(-0.25, 0.50), ObjectStart(0.35, 0.72), ObjectStart(-0.35, 0.55)],
    ),
    PushTestConfig(
        name="Multi-Object (4 extra)",
        category="multi_4", test_id=10,
        main_start=ObjectStart(0.10, 0.75),
        main_goal_x=-0.20, main_goal_y=0.50, main_goal_yaw_deg=90,
        extra_starts=[ObjectStart(-0.30, 0.60), ObjectStart(0.25, 0.55), ObjectStart(-0.35, 0.50), ObjectStart(0.35, 0.70)],
    ),
]

# =========================================================================
# 3. T-Push Tests (indices 41-50)
# =========================================================================
T_PUSH_TESTS = [
    PushTestConfig(
        name="T-Push",
        category="tpush", test_id=1, is_t_block=True,
        main_start=ObjectStart(-0.25, 0.70, 0),
        main_goal_x=0.25, main_goal_y=0.55, main_goal_yaw_deg=0,
        end_zone_x=0.05, end_zone_y=0.30,
    ),
    PushTestConfig(
        name="T-Push",
        category="tpush", test_id=2, is_t_block=True,
        main_start=ObjectStart(0.25, 0.60, 45),
        main_goal_x=-0.25, main_goal_y=0.70, main_goal_yaw_deg=90,
        end_zone_x=0.05, end_zone_y=0.30,
    ),
    PushTestConfig(
        name="T-Push",
        category="tpush", test_id=3, is_t_block=True,
        main_start=ObjectStart(-0.30, 0.55, 90),
        main_goal_x=0.30, main_goal_y=0.65, main_goal_yaw_deg=180,
        end_zone_x=0.05, end_zone_y=0.30,
    ),
    PushTestConfig(
        name="T-Push",
        category="tpush", test_id=4, is_t_block=True,
        main_start=ObjectStart(0.30, 0.70, 0),
        main_goal_x=-0.30, main_goal_y=0.55, main_goal_yaw_deg=-90,
        end_zone_x=0.05, end_zone_y=0.30,
    ),
    PushTestConfig(
        name="T-Push",
        category="tpush", test_id=5, is_t_block=True,
        main_start=ObjectStart(-0.20, 0.50, 180),
        main_goal_x=0.20, main_goal_y=0.75, main_goal_yaw_deg=45,
        end_zone_x=0.05, end_zone_y=0.30,
    ),
    PushTestConfig(
        name="T-Push",
        category="tpush", test_id=6, is_t_block=True,
        main_start=ObjectStart(0.20, 0.75, -45),
        main_goal_x=-0.20, main_goal_y=0.50, main_goal_yaw_deg=0,
        end_zone_x=0.05, end_zone_y=0.30,
    ),
    PushTestConfig(
        name="T-Push",
        category="tpush", test_id=7, is_t_block=True,
        main_start=ObjectStart(-0.35, 0.65, 90),
        main_goal_x=0.35, main_goal_y=0.55, main_goal_yaw_deg=-90,
        end_zone_x=0.05, end_zone_y=0.30,
    ),
    PushTestConfig(
        name="T-Push",
        category="tpush", test_id=8, is_t_block=True,
        main_start=ObjectStart(0.35, 0.55, 0),
        main_goal_x=-0.35, main_goal_y=0.65, main_goal_yaw_deg=180,
        end_zone_x=0.05, end_zone_y=0.30,
    ),
    PushTestConfig(
        name="T-Push",
        category="tpush", test_id=9, is_t_block=True,
        main_start=ObjectStart(0.00, 0.50, 0),
        main_goal_x=0.00, main_goal_y=0.70, main_goal_yaw_deg=0,
        end_zone_x=0.05, end_zone_y=0.30,
    ),
    PushTestConfig(
        name="T-Push",
        category="tpush", test_id=10, is_t_block=True,
        main_start=ObjectStart(0.00, 0.70, 0),
        main_goal_x=0.00, main_goal_y=0.50, main_goal_yaw_deg=90,
        end_zone_x=0.05, end_zone_y=0.30,
    ),
]

# =========================================================================
# Master list: index 0 = free-play, indices 1-50 = tests
# =========================================================================
ALL_TESTS: List[PushTestConfig] = [None]  # index 0 = free-play (no config)

for test in SINGLE_PUSH_TESTS:
    ALL_TESTS.append(test)
for test in MULTI_2_TESTS:
    ALL_TESTS.append(test)
for test in MULTI_3_TESTS:
    ALL_TESTS.append(test)
for test in MULTI_4_TESTS:
    ALL_TESTS.append(test)
for test in T_PUSH_TESTS:
    ALL_TESTS.append(test)


def get_test_count() -> int:
    """Return total number of test scenes (excluding index 0 free-play)."""
    return len(ALL_TESTS) - 1


def get_test_config(index: int) -> PushTestConfig:
    """Get test config by 0-based index (0 = free-play)."""
    if index < 0 or index >= len(ALL_TESTS):
        return ALL_TESTS[0]
    return ALL_TESTS[index]


def get_test_label(index: int) -> str:
    """Return a human-readable label for a test index."""
    if index == 0:
        return "Free-play"
    cfg = ALL_TESTS[index]
    if cfg is None:
        return "Free-play"
    return f"Test {index}/{get_test_count()} — {cfg.name} #{cfg.test_id}"
