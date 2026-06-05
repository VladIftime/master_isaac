#!/usr/bin/env python3
"""Convert dual-arm robot URDF to a self-contained USD in meshes/."""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch  # noqa: F401
from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
app = AppLauncher(parser.parse_args([]))
simulation_app = app.app

from isaaclab.sim.converters import UrdfConverter
from isaaclab.sim.converters.urdf_converter_cfg import UrdfConverterCfg

URDF = os.path.join(_PROJECT_ROOT, "urdf", "dual_arm_robot_rackets.urdf")
OUTPUT = os.path.join(_PROJECT_ROOT, "meshes")
os.makedirs(OUTPUT, exist_ok=True)

print(f"Converting: {URDF}  ->  {OUTPUT}/dual_arm_robot.usd", flush=True)

UrdfConverter(
    UrdfConverterCfg(
        asset_path=URDF,
        usd_dir=OUTPUT,
        usd_file_name="dual_arm_robot.usd",
        force_usd_conversion=True,
        make_instanceable=False,
        fix_base=False,
        merge_fixed_joints=False,
        self_collision=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=1000.0,
                damping=50.0,
            ),
        ),
    )
)

print("Done.", flush=True)
simulation_app.close()
