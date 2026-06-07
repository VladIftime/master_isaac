#!/usr/bin/env python3
"""Pre-bake drink001 USD: apply MassAPI to root and CollisionAPI to all meshes.

Usage:
    source /home/vladi/IsaacLab/master_isaac/.master_venv/bin/activate
    cd throwing_enviroment
    python scripts/prebake_drink.py
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
import shutil

NEW_USDS = os.path.join(_PROJECT_ROOT, "assets", "new_usds")
SRC = os.path.join(NEW_USDS, "drink001", "model_drink001.usd")
DST = os.path.join(NEW_USDS, "drink001", "drink_target.usd")

if os.path.exists(DST):
    os.remove(DST)
shutil.copy2(SRC, DST)
print(f"Copied to: {DST}", flush=True)

stage = Usd.Stage.Open(DST)
root = stage.GetDefaultPrim()

# Apply MassAPI to root
mass_api = UsdPhysics.MassAPI.Apply(root)
if mass_api:
    print(f"  MassAPI applied to root", flush=True)

# Apply CollisionAPI to all meshes
mesh_count = 0
for prim in stage.TraverseAll():
    if prim.IsA(UsdGeom.Mesh):
        coll = UsdPhysics.CollisionAPI.Apply(prim)
        if coll:
            coll.GetCollisionEnabledAttr().Set(True)
            mesh_count += 1
        try:
            physx_coll = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            if physx_coll:
                physx_coll.GetApproximationAttr().Set("convexDecomposition")
        except Exception:
            pass
        # Apply high-friction material for physics grip
        try:
            physx_mat = PhysxSchema.PhysxMaterialAPI.Apply(prim)
            if physx_mat:
                physx_mat.GetStaticFrictionAttr().Set(5.0)
                physx_mat.GetDynamicFrictionAttr().Set(5.0)
                physx_mat.GetRestitutionAttr().Set(0.1)
        except Exception:
            pass

print(f"  CollisionAPI applied to {mesh_count} meshes", flush=True)

stage.GetRootLayer().Save()
print(f"Saved: {DST}", flush=True)
print("Done.", flush=True)
simulation_app.close()
