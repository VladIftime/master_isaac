#!/usr/bin/env python3
"""Convert throwing environment OBJ meshes to instanceable USD with physics properties.

Runs inside the Isaac Sim/Isaac Lab kernel via AppLauncher.
Converts milk.obj, wooden_box.obj, and obstacle_box.obj to USD files with
mass, collision, and rigid body properties baked in.

Usage:
    source /home/vladi/IsaacLab/master_isaac/.master_venv/bin/activate
    cd throwing_enviroment
    python scripts/convert_meshes.py
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch._dynamo  # noqa: F401
import torch._C  # noqa: F401
import torch.optim  # noqa: F401

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
app = AppLauncher(parser.parse_args([]))
simulation_app = app.app

from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg

MESHES = os.path.join(_PROJECT_ROOT, "meshes")
OUTPUT = os.path.join(_PROJECT_ROOT, "generated_usd")
os.makedirs(OUTPUT, exist_ok=True)

conversions = [
    {
        "obj": os.path.join(MESHES, "milk.obj"),
        "usd": "milk.usd",
        "mass": 0.2,
        "kinematic": False,
        "scale": (0.001, 0.001, 0.001),
    },
    {
        "obj": os.path.join(MESHES, "wooden_box.obj"),
        "usd": "wooden_box.usd",
        "mass": 0.3,
        "kinematic": True,
        "scale": (0.001, 0.001, 0.001),
    },
    {
        "obj": os.path.join(MESHES, "obstacle_box.obj"),
        "usd": "obstacle_box.usd",
        "mass": 0.15,
        "kinematic": False,
        "scale": (0.001, 0.001, 0.001),
    },
]

for item in conversions:
    obj_path = item["obj"]
    if not os.path.exists(obj_path):
        print(f"MISSING: {obj_path}", flush=True)
        continue

    out_path = os.path.join(OUTPUT, item["usd"])
    print(f"Converting: {obj_path} -> {out_path}", flush=True)

    rigid_props = schemas_cfg.RigidBodyPropertiesCfg(
        kinematic_enabled=item["kinematic"],
        disable_gravity=item["kinematic"],
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=0,
        sleep_threshold=0.005,
        stabilization_threshold=0.0025,
        max_depenetration_velocity=1000.0,
    )

    collision_props = schemas_cfg.CollisionPropertiesCfg(
        collision_enabled=True,
    )

    mass_props = schemas_cfg.MassPropertiesCfg(mass=item["mass"])

    cfg = MeshConverterCfg(
        asset_path=obj_path,
        usd_dir=OUTPUT,
        usd_file_name=item["usd"],
        make_instanceable=False,
        mass_props=mass_props,
        rigid_props=rigid_props,
        collision_props=collision_props,
        scale=item["scale"],
    )

    converter = MeshConverter(cfg=cfg)
    print(f"  Done -> {converter.usd_path}", flush=True)

print("\nAll conversions complete.")
simulation_app.close()
