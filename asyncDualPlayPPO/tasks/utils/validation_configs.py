"""
Predefined test configurations for push model validation.

10 disc translation tests (position-only, disc object) +
10 T-block position-only tests (goal_yaw=0) +
10 T-block position+rotation tests (goal_yaw != 0).
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
    test_type: str          # "disc_pos", "pos_only", or "pos_rot"
    object_type: str = "tblock"  # "tblock" or "disc"
    main_start: StartPos = field(default_factory=lambda: StartPos(0.0, 0.0))
    main_goal_x: float = 0.0
    main_goal_y: float = 0.0
    main_goal_yaw: float = 0.0


ALL_TESTS: List[PushTestConfig] = [
    # ═══════════════════════════════════════════════════════════════════════
    # DISC — PURE TRANSLATION (position-only, disc object)
    # ═══════════════════════════════════════════════════════════════════════
    PushTestConfig("D_Forward",  1, "easy",   "disc_pos", "disc", StartPos(+0.00,+0.35), +0.00,+0.55, 0.0),
    PushTestConfig("D_Right",    2, "easy",   "disc_pos", "disc", StartPos(-0.15,+0.48), +0.05,+0.48, 0.0),
    PushTestConfig("D_Left",     3, "easy",   "disc_pos", "disc", StartPos(+0.15,+0.48), -0.05,+0.48, 0.0),
    PushTestConfig("D_Diag_FR",  4, "medium", "disc_pos", "disc", StartPos(-0.10,+0.38), +0.10,+0.58, 0.0),
    PushTestConfig("D_Diag_FL",  5, "medium", "disc_pos", "disc", StartPos(+0.10,+0.38), -0.10,+0.58, 0.0),
    PushTestConfig("D_Back",     6, "medium", "disc_pos", "disc", StartPos(+0.00,+0.60), +0.00,+0.35, 0.0),
    PushTestConfig("D_Wide_R",   7, "hard",   "disc_pos", "disc", StartPos(-0.30,+0.48), +0.10,+0.48, 0.0),
    PushTestConfig("D_Wide_L",   8, "hard",   "disc_pos", "disc", StartPos(+0.30,+0.48), -0.10,+0.48, 0.0),
    PushTestConfig("D_Long_F",   9, "hard",   "disc_pos", "disc", StartPos(+0.00,+0.30), +0.00,+0.65, 0.0),
    PushTestConfig("D_Cross",   10, "hard",   "disc_pos", "disc", StartPos(-0.20,+0.60), +0.20,+0.35, 0.0),

    # ═══════════════════════════════════════════════════════════════════════
    # T-BLOCK — POSITION-ONLY (goal_yaw=0 — object starts at yaw=0 too)
    # ═══════════════════════════════════════════════════════════════════════
    PushTestConfig("E_Forward", 11, "easy",   "pos_only", "tblock", StartPos(+0.00,+0.40), +0.00,+0.50, 0.0),
    PushTestConfig("E_Right",   12, "easy",   "pos_only", "tblock", StartPos(-0.10,+0.50), +0.00,+0.50, 0.0),
    PushTestConfig("E_Left",    13, "easy",   "pos_only", "tblock", StartPos(+0.10,+0.50), +0.00,+0.50, 0.0),
    PushTestConfig("M_Forward", 14, "medium", "pos_only", "tblock", StartPos(+0.00,+0.35), +0.00,+0.65, 0.0),
    PushTestConfig("M_Right",   15, "medium", "pos_only", "tblock", StartPos(-0.25,+0.50), +0.00,+0.50, 0.0),
    PushTestConfig("M_Left",    16, "medium", "pos_only", "tblock", StartPos(+0.25,+0.50), +0.00,+0.50, 0.0),
    PushTestConfig("H_Forward", 17, "hard",   "pos_only", "tblock", StartPos(+0.00,+0.30), +0.00,+0.70, 0.0),
    PushTestConfig("H_Right",   18, "hard",   "pos_only", "tblock", StartPos(-0.35,+0.50), +0.10,+0.50, 0.0),
    PushTestConfig("H_Left",    19, "hard",   "pos_only", "tblock", StartPos(+0.35,+0.50), -0.10,+0.50, 0.0),
    PushTestConfig("H_Cross",  20, "hard",   "pos_only", "tblock", StartPos(-0.25,+0.60), +0.25,+0.40, 0.0),

    # ═══════════════════════════════════════════════════════════════════════
    # T-BLOCK — POSITION + ROTATION (goal_yaw != 0)
    # ═══════════════════════════════════════════════════════════════════════
    PushTestConfig("E_Diag",    21, "easy",   "pos_rot", "tblock", StartPos(-0.10,+0.40), +0.00,+0.50, +0.52),
    PushTestConfig("M_Diag_R",  22, "medium", "pos_rot", "tblock", StartPos(-0.15,+0.35), +0.10,+0.60, +1.57),
    PushTestConfig("M_Diag_L",  23, "medium", "pos_rot", "tblock", StartPos(+0.15,+0.35), -0.10,+0.60, -1.05),
    PushTestConfig("M_Back",    24, "medium", "pos_rot", "tblock", StartPos(-0.10,+0.60), +0.10,+0.40, -2.09),
    PushTestConfig("H_Diag_FR", 25, "hard",   "pos_rot", "tblock", StartPos(-0.25,+0.35), +0.15,+0.65, +3.14),
    PushTestConfig("H_Diag_FL", 26, "hard",   "pos_rot", "tblock", StartPos(+0.25,+0.35), -0.15,+0.65, +0.79),
    PushTestConfig("H_Wide_R",  27, "hard",   "pos_rot", "tblock", StartPos(+0.40,+0.50), -0.40,+0.50, -3.14),
    PushTestConfig("H_Wide_L",  28, "hard",   "pos_rot", "tblock", StartPos(-0.40,+0.50), +0.40,+0.50, +2.36),
    PushTestConfig("Edge_Near", 29, "hard",   "pos_rot", "tblock", StartPos(-0.10,+0.65), +0.35,+0.35, -1.57),
    PushTestConfig("Edge_Far",  30, "hard",   "pos_rot", "tblock", StartPos(+0.35,+0.65), -0.35,+0.35, +2.62),
]


def get_test_config(test_index: int) -> PushTestConfig:
    if 1 <= test_index <= len(ALL_TESTS):
        return ALL_TESTS[test_index - 1]
    return None


def get_test_count() -> int:
    return len(ALL_TESTS)
