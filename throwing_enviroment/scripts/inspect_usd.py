#!/usr/bin/env python3
"""Inspect exported USD scene files for physics schemas and geometry info.

Usage:
    source /home/vladi/IsaacLab/master_isaac/.master_venv/bin/activate
    cd throwing_enviroment
    python scripts/inspect_usd.py
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

from pxr import Usd, UsdGeom, UsdPhysics

ASSETS_DIR = os.path.join(_PROJECT_ROOT, "assets")

usd_files = {
    "milk_carton": os.path.join(ASSETS_DIR, "milk", "meshes", "milk_carton_scene.usd"),
    "wooden_box": os.path.join(ASSETS_DIR, "wooden_box", "meshes", "box_scene.usd"),
    "milk_model": os.path.join(ASSETS_DIR, "milk", "meshes", "model.usd"),
    "wooden_box_model": os.path.join(ASSETS_DIR, "wooden_box", "meshes", "model.usd"),
}

for name, path in usd_files.items():
    print(f"\n{'='*60}", flush=True)
    print(f"Inspecting: {name}", flush=True)
    print(f"  Path: {path}", flush=True)
    print(f"  Exists: {os.path.exists(path)}", flush=True)
    if not os.path.exists(path):
        continue

    stage = Usd.Stage.Open(path)
    root = stage.GetDefaultPrim()
    if not root:
        root = stage.GetPseudoRoot()

    print(f"  Default/Root prim path: {root.GetPath()}", flush=True)
    print(f"  Prim type: {root.GetTypeName()}", flush=True)

    def inspect_prim(prim, depth=0):
        indent = "    " + "  " * depth
        prim_path_str = str(prim.GetPath())
        ptype = prim.GetTypeName()

        applied_schemas = [str(s) for s in prim.GetAppliedSchemas()]

        is_mesh = bool(UsdGeom.Mesh(prim))

        if is_mesh or depth == 0 or any(
            k in str(s)
            for k in ["RigidBody", "Collision", "Mass", "Physics"]
            for s in applied_schemas
        ):
            print(f"{indent}Prim: {prim_path_str}  (type={ptype})", flush=True)

        physics_schemas = [
            s for s in applied_schemas
            if any(k in s for k in ["RigidBody", "Collision", "Mass", "Physics"])
        ]
        if physics_schemas:
            print(f"{indent}  Physics schemas: {physics_schemas}", flush=True)

        if is_mesh:
            mesh = UsdGeom.Mesh(prim)
            points_attr = mesh.GetPointsAttr().Get()
            if points_attr:
                pts = list(points_attr)
                min_pt = [min(p[i] for p in pts) for i in range(3)]
                max_pt = [max(p[i] for p in pts) for i in range(3)]
                extent = [max_pt[i] - min_pt[i] for i in range(3)]
                print(f"{indent}  MESH: {len(pts)} vertices", flush=True)
                print(
                    f"{indent}    Extent: [{extent[0]:.4f}, {extent[1]:.4f}, {extent[2]:.4f}]",
                    flush=True,
                )

        purpose_attr = prim.GetAttribute("purpose")
        if purpose_attr:
            p = purpose_attr.Get()
            if p:
                print(f"{indent}  Purpose: {p}", flush=True)

        for child in prim.GetChildren():
            inspect_prim(child, depth + 1)

    inspect_prim(root)

print("\nDone.", flush=True)
simulation_app.close()
