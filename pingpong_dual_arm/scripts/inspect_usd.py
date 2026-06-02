#!/usr/bin/env python3
"""Inspect UR10_instanceable_pong.usd and print joint/link structure.

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

from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf, PhysxSchema

USD_PATH = os.path.join(_PROJECT_ROOT, "urdf", "UR10_instanceable_pong.usd")
stage = Usd.Stage.Open(USD_PATH)

print("=== PRIM HIERARCHY ===")
prim = stage.GetDefaultPrim()
root_path = str(prim.GetPath())

def print_tree(prim, indent=0):
    path = str(prim.GetPath())
    ptype = prim.GetTypeName()
    
    info = []
    
    if prim.IsA(UsdPhysics.Joint):
        info.append("(Joint)")
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        info.append("(RigidBody)")
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        info.append("(ArticulationRoot)")
    
    if prim.IsA(UsdGeom.Xformable):
        xform = UsdGeom.Xformable(prim)
        ops = xform.GetOrderedXformOps()
        if ops:
            for op in ops:
                op_name = op.GetOpName()
                if op_name == "xformOp:translate":
                    info.append(f"T={op.Get()}")
                elif op_name == "xformOp:orient":
                    info.append(f"O={op.Get()}")
    
    info_str = " ".join(info) if info else ""
    marker = " [JOINT]" if prim.IsA(UsdPhysics.Joint) else ""
    marker += " [RB]" if prim.HasAPI(UsdPhysics.RigidBodyAPI) else ""
    marker += " [ROOT]" if prim.HasAPI(UsdPhysics.ArticulationRootAPI) else ""
    print(f"{'  ' * indent}{prim.GetName()} [{ptype}]{marker} {info_str}")
    
    for child in prim.GetChildren():
        print_tree(child, indent + 1)

print_tree(prim)

# Also print all articulation root relationships
print("\n=== ARTICULATION ROOT ===")
for prim in stage.TraverseAll():
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        print(f"ArticulationRoot: {prim.GetPath()}")

print("\n=== ALL JOINTS ===")
for prim in stage.TraverseAll():
    if prim.IsA(UsdPhysics.Joint):
        joint = UsdPhysics.Joint(prim)
        path = str(prim.GetPath())
        attrs = {a.GetName(): a.Get() for a in prim.GetAttributes() if a.HasValue()}
        body0 = [str(t) for t in joint.GetBody0Rel().GetTargets()]
        body1 = [str(t) for t in joint.GetBody1Rel().GetTargets()]
        print(f"\nJoint: {path}")
        print(f"  Body0: {body0}")
        print(f"  Body1: {body1}")
        for k, v in attrs.items():
            if k in ("physics:axis", "physics:lowerLimit", "physics:upperLimit",
                     "xformOp:translate", "xformOp:orient"):
                print(f"  {k}: {v}")

print("\n=== LINKS (rigid bodies) ===")
for prim in stage.TraverseAll():
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        path = str(prim.GetPath())
        mass_api = UsdPhysics.MassAPI(prim)
        mass = mass_api.GetMassAttr().Get() if mass_api.GetMassAttr().HasValue() else "N/A"
        print(f"Link: {path} mass={mass}")
        # Print any visual mesh references
        for child in prim.GetChildren():
            if child.GetTypeName() == "Mesh":
                mesh = UsdGeom.Mesh(child)
                # check for mesh source
                if mesh.GetPointsAttr().HasValue():
                    pass  # inline points
            elif "visual" in str(child.GetPath()).lower() or "collision" in str(child.GetPath()).lower():
                for subchild in child.GetChildren():
                    if subchild.GetTypeName() == "Mesh":
                        mesh = UsdGeom.Mesh(subchild)
                        rel = mesh.GetPrim().GetRelationship("material:binding")
                        print(f"  Mesh child: {subchild.GetPath()}")

simulation_app.close()
