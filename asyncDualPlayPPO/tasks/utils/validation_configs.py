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
    # T-BLOCK — ROTATION-HEAVY (large yaw changes, mixed translation)
    # ═══════════════════════════════════════════════════════════════════════
    PushTestConfig("R_Pivot90",   1, "easy",   "pos_rot", "tblock", StartPos(+0.00,+0.50), +0.03,+0.53, +1.57),
    PushTestConfig("R_HalfTurn",  2, "easy",   "pos_rot", "tblock", StartPos(+0.00,+0.45), +0.00,+0.55, +3.14),
    PushTestConfig("R_QuarterL",  3, "medium", "pos_rot", "tblock", StartPos(+0.15,+0.55), -0.10,+0.50, +1.05),
    PushTestConfig("R_QuarterR",  4, "medium", "pos_rot", "tblock", StartPos(-0.15,+0.45), +0.10,+0.50, -0.79),
    PushTestConfig("R_DiagTurn",  5, "medium", "pos_rot", "tblock", StartPos(-0.12,+0.42), +0.12,+0.58, +2.09),
    PushTestConfig("R_Reverse",   6, "medium", "pos_rot", "tblock", StartPos(+0.00,+0.60), +0.00,+0.35, +1.57),
    PushTestConfig("R_SpinCross", 7, "hard",   "pos_rot", "tblock", StartPos(-0.30,+0.40), +0.30,+0.65, -2.62),
    PushTestConfig("R_WideSpin",  8, "hard",   "pos_rot", "tblock", StartPos(+0.35,+0.55), -0.35,+0.50, +2.36),
    PushTestConfig("R_LongSpin",  9, "hard",   "pos_rot", "tblock", StartPos(+0.00,+0.30), +0.00,+0.70, +3.00),
    PushTestConfig("R_Corner",   10, "hard",   "pos_rot", "tblock", StartPos(-0.30,+0.60), +0.30,+0.30, -1.57),

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

    # ═══════════════════════════════════════════════════════════════════════
    # DEMO SCENES — object already within success thresholds (static hold)
    # ═══════════════════════════════════════════════════════════════════════
    PushTestConfig("Demo_Pos",    31, "demo", "demo", "tblock", StartPos(+0.05,+0.50), +0.00,+0.50, 0.0),
    PushTestConfig("Demo_PosRot", 32, "demo", "demo", "tblock", StartPos(+0.05,+0.50), +0.00,+0.50, 0.2),
]


# ═══════════════════════════════════════════════════════════════════════════════
# DISC test set — the 30 T-block scenes mirrored onto the rotationally-symmetric
# disc: identical start/goal positions and difficulty, but goal_yaw forced to 0
# (orientation always passes) and object_type="disc", test_type="disc_pos".
# Used by the ABC-on disc positive control (phase4_pc) so a disc-trained Bob is
# evaluated on disc scenes, not the T-block scenes it never saw.
# ═══════════════════════════════════════════════════════════════════════════════
DISC_TESTS: List[PushTestConfig] = [
    PushTestConfig(f"D_{t.name}", t.test_id, t.difficulty, "disc_pos", "disc",
                   StartPos(t.main_start.x, t.main_start.y),
                   t.main_goal_x, t.main_goal_y, 0.0)
    for t in ALL_TESTS[:30]
]


# Active test set (switchable via set_test_set); defaults to the T-block suite.
_TEST_SETS = {"all": ALL_TESTS, "tblock": ALL_TESTS, "disc": DISC_TESTS}
_ACTIVE_TESTS: List[PushTestConfig] = ALL_TESTS


def set_test_set(name: str) -> None:
    """Select which scene list get_test_config/get_test_count return."""
    global _ACTIVE_TESTS
    if name not in _TEST_SETS:
        raise ValueError(f"unknown scene set '{name}' (choose from {list(_TEST_SETS)})")
    _ACTIVE_TESTS = _TEST_SETS[name]


def get_test_config(test_index: int) -> PushTestConfig:
    if 1 <= test_index <= len(_ACTIVE_TESTS):
        return _ACTIVE_TESTS[test_index - 1]
    return None


def get_test_count() -> int:
    return len(_ACTIVE_TESTS)
