#!/usr/bin/env python3
"""Pre-bake the dual-arm robot URDF to USD for faster scene creation.

Runs the URDF-to-USD conversion once and saves the result to
assets/robot/dual_arm_robot.usd. Subsequent env creation will load
the pre-built USD directly (no runtime conversion needed).

Usage:
    source /home/vladi/IsaacLab/master_isaac/.master_venv/bin/activate
    cd throwing_enviroment
    python scripts/prebake_robot_usd.py
"""

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

URDF_PATH = os.path.join(_PROJECT_ROOT, "urdf", "dual_arm_robot.urdf")
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "assets", "robot")
OUTPUT_USD = os.path.join(OUTPUT_DIR, "dual_arm_robot.usd")


def main():
    if not os.path.exists(URDF_PATH):
        print(f"[ERROR] URDF not found: {URDF_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[INFO] Converting URDF to USD...")
    print(f"  Input:  {URDF_PATH}")
    print(f"  Output: {OUTPUT_USD}")

    cfg = UrdfConverterCfg(
        asset_path=URDF_PATH,
        usd_dir=OUTPUT_DIR,
        usd_file_name="dual_arm_robot.usd",
        fix_base=False,
        make_instanceable=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=1000.0,
                damping=50.0,
            ),
        ),
    )

    converter = UrdfConverter(cfg)
    usd_path = converter.usd_path

    print(f"[INFO] Conversion complete!")
    print(f"  USD saved to: {usd_path}")
    print(f"  File size: {os.path.getsize(usd_path) / 1024:.1f} KB")
    print()
    print(f"  The DirectRLEnv will now automatically use this USD")
    print(f"  (detected at: assets/robot/dual_arm_robot.usd)")


if __name__ == "__main__":
    main()
    simulation_app.close()
