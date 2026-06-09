#!/usr/bin/env python3
"""Pre-bake trash can USD: open lid, set pedal pressed, apply mass/physics, make kinematic.

The trash can from Synthesis-Assets-Explorer has articulated lid + pedal joints.
For use as a kinematic target object, we:
  1. Rotate the lid prim to -90° (fully open)
  2. Rotate the pedal prim to 30° (pressed down)
  3. Apply RigidBodyAPI to root, set all bodies to kinematic
  4. Apply CollisionAPI to all meshes
  5. Set mass on each rigid body
  6. Save as a modified copy

Usage:
    source /home/vladi/IsaacLab/master_isaac/.master_venv/bin/activate
    cd throwing_enviroment
    python scripts/prebake_trash_can.py
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

from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf
import shutil

NEW_USDS = os.path.join(_PROJECT_ROOT, "assets", "new_usds")
SRC = os.path.join(NEW_USDS, "trash can002", "model_stepping_dustbin_4.usd")
DST = os.path.join(NEW_USDS, "trash can002", "trash_can_open.usd")

# Copy to a new file so we don't modify the original
if os.path.exists(DST):
    os.remove(DST)
shutil.copy2(SRC, DST)
print(f"Copied: {SRC} -> {DST}", flush=True)

stage = Usd.Stage.Open(DST)
root = stage.GetDefaultPrim()
if not root:
    print("ERROR: no default prim", flush=True)
    simulation_app.close()
    exit(1)

print(f"Root prim: {root.GetPath()}  type={root.GetTypeName()}", flush=True)

# Find lid, body, pedal prims
lid_prim = None
body_prim = None
pedal_prim = None

for prim in stage.TraverseAll():
    path = str(prim.GetPath())
    if path.endswith("E_lid_1") and prim.IsA(UsdGeom.Xformable):
        lid_prim = prim
    elif path.endswith("E_body_2") and prim.IsA(UsdGeom.Xformable):
        body_prim = prim
    elif path.endswith("E_pedal_5") and prim.IsA(UsdGeom.Xformable):
        pedal_prim = prim

if not all([lid_prim, body_prim, pedal_prim]):
    print(
        f"ERROR: Missing prims: lid={lid_prim}, body={body_prim}, pedal={pedal_prim}",
        flush=True,
    )
    simulation_app.close()
    exit(1)

print(f"Lid:   {lid_prim.GetPath()}", flush=True)
print(f"Body:  {body_prim.GetPath()}", flush=True)
print(f"Pedal: {pedal_prim.GetPath()}", flush=True)

# Apply RigidBodyAPI to root and make all rigid bodies kinematic
root_rb = UsdPhysics.RigidBodyAPI.Apply(root)
if root_rb:
    root_rb.GetKinematicEnabledAttr().Set(True)
    print("Applied RigidBodyAPI (kinematic) to root", flush=True)

for prim in stage.TraverseAll():
    rb = UsdPhysics.RigidBodyAPI(prim)
    if rb:
        rb.GetKinematicEnabledAttr().Set(True)
        print(f"  Set kinematic on {prim.GetPath()}", flush=True)

# Apply CollisionAPI to all meshes
mesh_count = 0
for prim in stage.TraverseAll():
    if prim.IsA(UsdGeom.Mesh):
        coll_api = UsdPhysics.CollisionAPI.Apply(prim)
        if coll_api:
            coll_api.GetCollisionEnabledAttr().Set(True)
            mesh_count += 1
            try:
                physx_coll = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                if physx_coll:
                    physx_coll.GetApproximationAttr().Set("convexDecomposition")
            except Exception:
                pass
print(f"CollisionAPI applied to {mesh_count} meshes", flush=True)

# Set mass on each rigid body
mass_values = {
    "E_lid_1": 0.3,
    "E_body_2": 1.5,
    "E_pedal_5": 0.2,
}
for prim in stage.TraverseAll():
    basename = prim.GetPath().name
    if basename in mass_values:
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        if mass_api:
            mass_api.GetMassAttr().Set(mass_values[basename])
            print(f"  Mass {basename}: {mass_values[basename]} kg", flush=True)


def rotate_around_x(prim, angle_deg, op_suffix):
    """Rotate a prim around its local X axis by composing with existing transform."""
    xf = UsdGeom.Xformable(prim)
    ops = xf.GetOrderedXformOps()

    if len(ops) == 1 and "transform" in ops[0].GetName():
        existing_mat = Gf.Matrix4d(ops[0].Get())
        import math

        angle_rad = angle_deg * math.pi / 180.0
        rotation = Gf.Rotation(Gf.Vec3d.XAxis(), angle_rad)
        new_mat = existing_mat * Gf.Matrix4d(rotation, Gf.Vec3d(0))
        xf.ClearXformOpOrder()
        new_op = xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, op_suffix)
        new_op.Set(new_mat)
    else:
        rot_op = xf.AddRotateXOp(UsdGeom.XformOp.PrecisionDouble, op_suffix)
        rot_op.Set(angle_deg)


rotate_around_x(lid_prim, -90.0, "open_lid")
print(f"Lid rotated -90° around X (open)", flush=True)

rotate_around_x(pedal_prim, 30.0, "press_pedal")
print(f"Pedal rotated 30° around X (pressed)", flush=True)

# Update joint drive targets to match
for prim in stage.TraverseAll():
    if "RevoluteJoint" in prim.GetTypeName():
        path = str(prim.GetPath())
        if "up" in path:
            prim.GetAttribute("drive:angular:physics:targetPosition").Set(-90.0)
            print(f"  Joint {path} target set to -90°", flush=True)
        elif "down" in path:
            prim.GetAttribute("drive:angular:physics:targetPosition").Set(30.0)
            print(f"  Joint {path} target set to 30°", flush=True)

stage.GetRootLayer().Save()
print(f"\nSaved: {DST}", flush=True)
print("Done.", flush=True)
simulation_app.close()
