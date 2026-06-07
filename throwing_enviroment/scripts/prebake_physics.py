#!/usr/bin/env python3
"""Pre-bake physics schemas into generated USD files.

Ensures RigidBodyAPI, CollisionAPI, and MassAPI are applied so that
UsdFileCfg can set physics properties at spawn time.

Usage:
    source /home/vladi/IsaacLab/master_isaac/.master_venv/bin/activate
    cd throwing_enviroment
    python scripts/prebake_physics.py
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

from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema

targets = [
    os.path.join(_PROJECT_ROOT, "generated_usd", "milk.usd"),
    os.path.join(_PROJECT_ROOT, "generated_usd", "wooden_box.usd"),
]

for usd_path in targets:
    name = os.path.basename(usd_path)

    if not os.path.exists(usd_path):
        print(f"SKIP {name}: not found", flush=True)
        continue

    print(f"\nProcessing: {name}", flush=True)
    print(f"  Path: {usd_path}", flush=True)

    stage = Usd.Stage.Open(usd_path)
    root = stage.GetDefaultPrim()
    if not root:
        print("  ERROR: No default prim", flush=True)
        continue

    print(f"  Root prim: {root.GetPath()}  type={root.GetTypeName()}", flush=True)

    UsdPhysics.RigidBodyAPI.Apply(root)
    print(f"  Applied RigidBodyAPI", flush=True)

    UsdPhysics.MassAPI.Apply(root)
    print(f"  Applied MassAPI", flush=True)

    mesh_count = 0
    collision_count = 0

    def apply_collision(prim):
        nonlocal mesh_count, collision_count
        if prim.IsA(UsdGeom.Mesh):
            mesh_count += 1
            api = UsdPhysics.CollisionAPI.Apply(prim)
            if api:
                collision_count += 1
                api.GetCollisionEnabledAttr().Set(True)
                try:
                    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                    if px:
                        px.GetApproximationAttr().Set("convexDecomposition")
                except Exception:
                    pass
        for child in prim.GetChildren():
            apply_collision(child)

    apply_collision(root)
    print(f"  Meshes: {mesh_count}  CollisionAPI: {collision_count}", flush=True)

    stage.GetRootLayer().Save()
    print(f"  Saved", flush=True)

print("\nDone.", flush=True)
simulation_app.close()
