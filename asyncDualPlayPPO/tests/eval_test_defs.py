"""
Evaluation test definitions for push-T model comparison.

10 test scenarios with deterministic seeded episode generation.
Tests 1-8: Fixed scenarios (20 episodes each, spawn noise).
Tests 9-10: Random difficulty buckets (100 episodes each).

All models receive identical (start, goal) pairs for fair comparison.
"""

import math
from dataclasses import dataclass
from typing import List

import torch


@dataclass
class EpisodeConfig:
    start_x: float
    start_y: float
    start_yaw: float
    goal_x: float
    goal_y: float
    goal_yaw: float


@dataclass
class TestDef:
    test_id: int
    name: str
    n_episodes: int
    description: str


WORKSPACE_X = (-0.35, 0.35)
WORKSPACE_Y = (0.30, 0.65)

TESTS: List[TestDef] = [
    TestDef(1,  "short_translation", 10,  "Pure translation 0.08m, no rotation"),
    TestDef(2,  "long_translation",  10,  "Pure translation 0.25m, no rotation"),
    TestDef(3,  "small_rotation",    10,  "Pure rotation 30 deg, no translation"),
    TestDef(4,  "large_rotation",    10,  "Pure rotation 90 deg, no translation"),
    TestDef(5,  "combined_easy",     10,  "Translation 0.10m + rotation 30 deg"),
    TestDef(6,  "combined_hard",     10,  "Translation 0.25m + rotation 90 deg"),
    TestDef(7,  "precision",         10,  "Small correction: 0.03m + 10 deg"),
    TestDef(8,  "boundary_push",     10,  "Goal near workspace edge"),
    TestDef(9,  "random_easy",       10, "Random: pos in [0.05,0.15]m, rot in [0,0.5]rad"),
    TestDef(10, "random_hard",       10, "Random: pos in [0.15,0.35]m, rot in [0.5,2.5]rad"),
]


def _clamp_pos(x: float, y: float) -> tuple:
    x = max(WORKSPACE_X[0], min(WORKSPACE_X[1], x))
    y = max(WORKSPACE_Y[0], min(WORKSPACE_Y[1], y))
    return x, y


def _wrap_yaw(yaw: float) -> float:
    while yaw > math.pi:
        yaw -= 2 * math.pi
    while yaw < -math.pi:
        yaw += 2 * math.pi
    return yaw


def generate_episodes(test_id: int, master_seed: int = 42) -> List[EpisodeConfig]:
    test_def = None
    for t in TESTS:
        if t.test_id == test_id:
            test_def = t
            break
    if test_def is None:
        raise ValueError(f"Unknown test_id: {test_id}")

    episodes = []
    gen = torch.Generator()
    gen.manual_seed(master_seed * 1000 + test_id * 100)

    if test_id <= 8:
        episodes = _generate_fixed_test(test_id, test_def.n_episodes, gen)
    elif test_id == 9:
        episodes = _generate_random_test(
            n=test_def.n_episodes, gen=gen,
            pos_range=(0.05, 0.15), rot_range=(0.0, 0.5),
        )
    elif test_id == 10:
        episodes = _generate_random_test(
            n=test_def.n_episodes, gen=gen,
            pos_range=(0.15, 0.35), rot_range=(0.5, 2.5),
        )

    return episodes


def _generate_fixed_test(test_id: int, n: int, gen: torch.Generator) -> List[EpisodeConfig]:
    base_x, base_y = 0.0, 0.50
    pos_noise_std = 0.02
    yaw_noise_std = 0.17

    if test_id == 1:
        delta_pos, delta_yaw = 0.08, 0.0
    elif test_id == 2:
        delta_pos, delta_yaw = 0.25, 0.0
    elif test_id == 3:
        delta_pos, delta_yaw = 0.0, math.radians(30)
    elif test_id == 4:
        delta_pos, delta_yaw = 0.0, math.radians(90)
    elif test_id == 5:
        delta_pos, delta_yaw = 0.10, math.radians(30)
    elif test_id == 6:
        delta_pos, delta_yaw = 0.25, math.radians(90)
    elif test_id == 7:
        delta_pos, delta_yaw = 0.03, math.radians(10)
        pos_noise_std = 0.01
        yaw_noise_std = 0.09
    elif test_id == 8:
        base_x, base_y = 0.0, 0.45
        delta_pos, delta_yaw = 0.20, math.radians(45)
        pos_noise_std = 0.02
        yaw_noise_std = 0.17
    else:
        delta_pos, delta_yaw = 0.10, 0.0

    episodes = []
    for i in range(n):
        direction = torch.rand(1, generator=gen).item() * 2 * math.pi

        noise_sx = torch.randn(1, generator=gen).item() * pos_noise_std
        noise_sy = torch.randn(1, generator=gen).item() * pos_noise_std
        noise_syaw = torch.randn(1, generator=gen).item() * yaw_noise_std

        start_x = base_x + noise_sx
        start_y = base_y + noise_sy
        start_yaw = _wrap_yaw(noise_syaw)
        start_x, start_y = _clamp_pos(start_x, start_y)

        goal_x = start_x + delta_pos * math.cos(direction)
        goal_y = start_y + delta_pos * math.sin(direction)

        if test_id == 8:
            edge_dir = torch.rand(1, generator=gen).item()
            if edge_dir < 0.25:
                goal_x = WORKSPACE_X[1] - 0.02
            elif edge_dir < 0.5:
                goal_x = WORKSPACE_X[0] + 0.02
            elif edge_dir < 0.75:
                goal_y = WORKSPACE_Y[1] - 0.02
            else:
                goal_y = WORKSPACE_Y[0] + 0.02
            goal_x = start_x + (goal_x - start_x) * 0.8
            goal_y = start_y + (goal_y - start_y) * 0.8

        goal_x, goal_y = _clamp_pos(goal_x, goal_y)

        yaw_sign = 1.0 if torch.rand(1, generator=gen).item() > 0.5 else -1.0
        goal_yaw = _wrap_yaw(start_yaw + yaw_sign * delta_yaw)

        episodes.append(EpisodeConfig(
            start_x=start_x, start_y=start_y, start_yaw=start_yaw,
            goal_x=goal_x, goal_y=goal_y, goal_yaw=goal_yaw,
        ))

    return episodes


def _generate_random_test(
    n: int,
    gen: torch.Generator,
    pos_range: tuple,
    rot_range: tuple,
) -> List[EpisodeConfig]:
    episodes = []
    attempts = 0
    max_attempts = n * 20

    while len(episodes) < n and attempts < max_attempts:
        attempts += 1

        start_x = WORKSPACE_X[0] + torch.rand(1, generator=gen).item() * (WORKSPACE_X[1] - WORKSPACE_X[0])
        start_y = WORKSPACE_Y[0] + torch.rand(1, generator=gen).item() * (WORKSPACE_Y[1] - WORKSPACE_Y[0])
        start_yaw = _wrap_yaw((torch.rand(1, generator=gen).item() - 0.5) * 2 * math.pi)

        dist = pos_range[0] + torch.rand(1, generator=gen).item() * (pos_range[1] - pos_range[0])
        angle = torch.rand(1, generator=gen).item() * 2 * math.pi
        goal_x = start_x + dist * math.cos(angle)
        goal_y = start_y + dist * math.sin(angle)

        goal_x, goal_y = _clamp_pos(goal_x, goal_y)

        actual_dist = math.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)
        if actual_dist < pos_range[0] * 0.8:
            continue

        rot_mag = rot_range[0] + torch.rand(1, generator=gen).item() * (rot_range[1] - rot_range[0])
        yaw_sign = 1.0 if torch.rand(1, generator=gen).item() > 0.5 else -1.0
        goal_yaw = _wrap_yaw(start_yaw + yaw_sign * rot_mag)

        episodes.append(EpisodeConfig(
            start_x=start_x, start_y=start_y, start_yaw=start_yaw,
            goal_x=goal_x, goal_y=goal_y, goal_yaw=goal_yaw,
        ))

    return episodes


def get_test_def(test_id: int) -> TestDef:
    for t in TESTS:
        if t.test_id == test_id:
            return t
    raise ValueError(f"Unknown test_id: {test_id}")
