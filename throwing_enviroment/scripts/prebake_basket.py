#!/usr/bin/env python3
"""Pre-bake shopping basket USD: strip to single rigid body for RigidObjectCfg.

The original model has 3 rigid bodies (body + 2 handles) with 2 revolute joints.
RigidObjectCfg requires exactly ONE rigid body. This script:
  1. Removes RigidBodyAPI from handles (they become static visual children)
  2. Removes RigidBodyAPI from root prim
  3. Keeps RigidBodyAPI only on E_body_20 (kinematic target)
  4. Deactivates joints (no longer needed)
  5. Applies CollisionAPI to all meshes
  6. Sets mass on E_body_20

Usage:
    source /home/vladi/IsaacLab/master_isaac/.master_venv/bin/activate
    cd throwing_enviroment
    python scripts/prebake_basket.py
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
SRC = os.path.join(NEW_USDS, "shopping basket002", "model_basket_22.usd")
DST = os.path.join(NEW_USDS, "shopping basket002", "basket_target.usd")

if os.path.exists(DST):
    os.remove(DST)
shutil.copy2(SRC, DST)
print(f"Copied to: {DST}", flush=True)

stage = Usd.Stage.Open(DST)
root = stage.GetDefaultPrim()
layer = stage.GetRootLayer()

# Ensure root has no RigidBodyAPI
if root.HasAPI(UsdPhysics.RigidBodyAPI):
    root.RemoveAPI(UsdPhysics.RigidBodyAPI)
    print("  Removed RigidBodyAPI from root", flush=True)

handle_names = {"E_bail_handle_01_25", "E_bail_handle_02_28"}

# Process joints FIRST (before deactivating parent handles)
for prim in list(stage.TraverseAll()):
    if prim.IsValid() and "Joint" in prim.GetTypeName() and prim.IsA(UsdPhysics.Joint):
        prim.SetActive(False)
        print(f"  Deactivated joint: {prim.GetPath()}", flush=True)

# Then process handles and articulation roots
for prim in list(stage.TraverseAll()):
    if not prim.IsValid():
        continue
    path = prim.GetPath()
    name = path.name

    if name in handle_names:
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            print(f"  Removed RigidBodyAPI from {path}", flush=True)
        prim.SetActive(False)
        print(f"  Deactivated handle: {path}", flush=True)

    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        print(f"  Removed ArticulationRootAPI from {path}", flush=True)

# Now set up E_body_20 as the sole kinematic rigid body
body_prim = stage.GetPrimAtPath("/root/E_body_20")
if body_prim and body_prim.IsValid():
    body_rb = UsdPhysics.RigidBodyAPI.Apply(body_prim)
    body_rb.GetKinematicEnabledAttr().Set(True)
    print("  E_body_20: kinematic rigid body", flush=True)

    mass_api = UsdPhysics.MassAPI.Apply(body_prim)
    mass_api.GetMassAttr().Set(2.0)
    print("  E_body_20: mass = 2.0 kg", flush=True)

# Apply CollisionAPI to all meshes
for prim in stage.TraverseAll():
    if prim.IsA(UsdGeom.Mesh):
        coll = UsdPhysics.CollisionAPI.Apply(prim)
        if coll:
            coll.GetCollisionEnabledAttr().Set(True)
        try:
            physx_coll = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            if physx_coll:
                physx_coll.GetApproximationAttr().Set("convexDecomposition")
        except Exception:
            pass

print(f"  CollisionAPI applied to all meshes", flush=True)

stage.GetRootLayer().Save()
print(f"Saved: {DST}", flush=True)
print("Done.", flush=True)
simulation_app.close()
