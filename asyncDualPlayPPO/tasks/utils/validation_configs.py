"""
Predefined test configurations for push model validation.

10 position-only tests (goal_yaw=0) + 10 position+rotation tests (goal_yaw != 0).
Each test defines object start position and goal position + yaw.
Difficulty categories: easy/medium/hard based on 2D distance from start to goal.
All coordinates in env-local frame (metres), yaw in radians.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class StartPos:
    x: float
    y: float


@dataclass
class PushTestConfig:
    name: str
    test_id: int
    difficulty: str
    test_type: str          # "pos_only" or "pos_rot"
    main_start: StartPos = field(default_factory=lambda: StartPos(0.0, 0.0))
    main_goal_x: float = 0.0
    main_goal_y: float = 0.0
    main_goal_yaw: float = 0.0


ALL_TESTS: List[PushTestConfig] = [
    # ═══════════════════════════════════════════════════════════════════════
    # POSITION-ONLY (goal_yaw=0 — object starts at yaw=0 too)
    # ═══════════════════════════════════════════════════════════════════════
    PushTestConfig("E_Forward",  1, "easy",   "pos_only", StartPos(+0.00,+0.40), +0.00,+0.50, 0.0),
    PushTestConfig("E_Right",    2, "easy",   "pos_only", StartPos(-0.10,+0.50), +0.00,+0.50, 0.0),
    PushTestConfig("E_Left",     3, "easy",   "pos_only", StartPos(+0.10,+0.50), +0.00,+0.50, 0.0),
    PushTestConfig("M_Forward",  4, "medium", "pos_only", StartPos(+0.00,+0.35), +0.00,+0.65, 0.0),
    PushTestConfig("M_Right",    5, "medium", "pos_only", StartPos(-0.25,+0.50), +0.00,+0.50, 0.0),
    PushTestConfig("M_Left",     6, "medium", "pos_only", StartPos(+0.25,+0.50), +0.00,+0.50, 0.0),
    PushTestConfig("H_Forward",  7, "hard",   "pos_only", StartPos(+0.00,+0.30), +0.00,+0.70, 0.0),
    PushTestConfig("H_Right",    8, "hard",   "pos_only", StartPos(-0.35,+0.50), +0.10,+0.50, 0.0),
    PushTestConfig("H_Left",     9, "hard",   "pos_only", StartPos(+0.35,+0.50), -0.10,+0.50, 0.0),
    PushTestConfig("H_Cross",   10, "hard",   "pos_only", StartPos(-0.25,+0.60), +0.25,+0.40, 0.0),

    # ═══════════════════════════════════════════════════════════════════════
    # POSITION + ROTATION (goal_yaw != 0)
    # ═══════════════════════════════════════════════════════════════════════
    PushTestConfig("E_Diag",    11, "easy",   "pos_rot", StartPos(-0.10,+0.40), +0.00,+0.50, +0.52),  # ~30 deg
    PushTestConfig("M_Diag_R",  12, "medium", "pos_rot", StartPos(-0.15,+0.35), +0.10,+0.60, +1.57),  # 90 deg
    PushTestConfig("M_Diag_L",  13, "medium", "pos_rot", StartPos(+0.15,+0.35), -0.10,+0.60, -1.05),  # -60 deg
    PushTestConfig("M_Back",    14, "medium", "pos_rot", StartPos(-0.10,+0.60), +0.10,+0.40, -2.09),  # -120 deg
    PushTestConfig("H_Diag_FR", 15, "hard",   "pos_rot", StartPos(-0.25,+0.35), +0.15,+0.65, +3.14),  # 180 deg
    PushTestConfig("H_Diag_FL", 16, "hard",   "pos_rot", StartPos(+0.25,+0.35), -0.15,+0.65, +0.79),  # 45 deg
    PushTestConfig("H_Wide_R",  17, "hard",   "pos_rot", StartPos(+0.40,+0.50), -0.40,+0.50, -3.14),  # -180 deg
    PushTestConfig("H_Wide_L",  18, "hard",   "pos_rot", StartPos(-0.40,+0.50), +0.40,+0.50, +2.36),  # 135 deg
    PushTestConfig("Edge_Near", 19, "hard",   "pos_rot", StartPos(-0.10,+0.65), +0.35,+0.35, -1.57),  # -90 deg
    PushTestConfig("Edge_Far",  20, "hard",   "pos_rot", StartPos(+0.35,+0.65), -0.35,+0.35, +2.62),  # 150 deg
]


def get_test_config(test_index: int) -> PushTestConfig:
    if 1 <= test_index <= len(ALL_TESTS):
        return ALL_TESTS[test_index - 1]
    return None


def get_test_count() -> int:
    return len(ALL_TESTS)
