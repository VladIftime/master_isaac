#!/usr/bin/env python3
"""Convert UR10_instanceable_pong.usd into a single-arm URDF with racket gripper.

Uses Isaac Sim's isaacsim.asset.exporter.urdf extension (UsdToUrdf) to
extract the robot articulation from the USD and write a URDF.

Usage:
    source /home/vladi/IsaacLab/master_isaac/.master_venv/bin/activate
    cd pingpong_dual_arm
    python scripts/convert_usd_to_urdf.py
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch._dynamo  # noqa: F401
import torch._C       # noqa: F401
import torch.optim    # noqa: F401

from isaaclab.app import AppLauncher
import argparse
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
app = AppLauncher(parser.parse_args([]))
simulation_app = app.app

import omni.kit.app

manager = omni.kit.app.get_app().get_extension_manager()
if not manager.is_extension_enabled("isaacsim.asset.exporter.urdf"):
    manager.set_extension_enabled_immediate("isaacsim.asset.exporter.urdf", True)

import logging
from nvidia.srl.from_usd.to_urdf import UsdToUrdf

USD_PATH = os.path.join(_PROJECT_ROOT, "urdf", "UR10_instanceable_pong.usd")
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "urdf")
OUTPUT_URDF = os.path.join(OUTPUT_DIR, "UR10_instanceable_pong.urdf")

print(f"Converting: {USD_PATH}", flush=True)
print(f"Output:     {OUTPUT_URDF}", flush=True)

usd_to_urdf_kwargs = {
    "node_names_to_remove": None,
    "edge_names_to_remove": None,
    "root": None,
    "parent_link_is_body_1": None,
    "log_level": logging.ERROR,
}

usd_to_urdf = UsdToUrdf.init_from_file(USD_PATH, **usd_to_urdf_kwargs)

MESH_DIR = os.path.join(OUTPUT_DIR, "meshes", "UR10_instanceable_pong")
os.makedirs(MESH_DIR, exist_ok=True)

usd_to_urdf.save_to_file(
    urdf_output_path=OUTPUT_URDF,
    visualize_collision_meshes=False,
    mesh_dir=MESH_DIR,
    mesh_path_prefix="",
)

# The USD->URDF converter produces "inf" effort limits which are invalid URDF.
# Replace them with 0 (no limit).
with open(OUTPUT_URDF) as f:
    content = f.read()

content = content.replace("inf", "0.")

with open(OUTPUT_URDF, "w") as f:
    f.write(content)

print(f"Done. URDF written to {OUTPUT_URDF}", flush=True)
print(f"Meshes extracted to {MESH_DIR}", flush=True)

simulation_app.close()
