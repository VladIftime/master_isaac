#!/usr/bin/env python3
"""Regenerate ping pong USD assets with proper physics tags and NO instancing.

MeshConverter with make_instanceable=False embeds the mesh directly in the
output USD so all APIs (RigidBody, Collision) are on the same prim tree.

Usage:
    source /home/vladi/IsaacLab/master_isaac/.master_venv/bin/activate
    cd pingpong_dual_arm
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

from isaaclab.sim.converters import MeshConverter
from isaaclab.sim.converters.mesh_converter_cfg import MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg

MESHES = os.path.join(_PROJECT_ROOT, "meshes", "pingpong")
OUTPUT = os.path.join(_PROJECT_ROOT, "assets", "pingpong")
os.makedirs(OUTPUT, exist_ok=True)

conversions = [
    {
        "stl": os.path.join(MESHES, "ping_pong_", "ping_pong_table", "table.stl"),
        "usd": "table.usd",
        "scale": (0.001, 0.001, 0.001),
        "kinematic": True,
    },
    {
        "stl": os.path.join(MESHES, "racket.stl"),
        "usd": "racket.usd",
        "scale": (0.001, 0.001, 0.001),
        "kinematic": True,
    },
]

for item in conversions:
    stl_path = item["stl"]
    if not os.path.exists(stl_path):
        print(f"MISSING: {stl_path}", flush=True)
        continue

    print(f"Converting: {stl_path}  ->  {OUTPUT}/{item['usd']}", flush=True)

    cfg_kwargs = dict(
        asset_path=stl_path,
        usd_dir=OUTPUT,
        usd_file_name=item["usd"],
        force_usd_conversion=True,
        make_instanceable=False,
        scale=item["scale"],
        rigid_props=schemas_cfg.RigidBodyPropertiesCfg(
            kinematic_enabled=item["kinematic"],
            disable_gravity=True,
        ),
        collision_props=schemas_cfg.CollisionPropertiesCfg(),
    )
    if "rotation" in item:
        cfg_kwargs["rotation"] = item["rotation"]

    MeshConverter(MeshConverterCfg(**cfg_kwargs))
    print(f"  Done", flush=True)

print("\nAll assets converted.", flush=True)
simulation_app.close()
