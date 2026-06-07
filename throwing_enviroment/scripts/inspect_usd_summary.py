#!/usr/bin/env python3
"""Quick inspection of exported USD scene files — summary per file."""

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

from pxr import Usd, UsdGeom

ASSETS_DIR = os.path.join(_PROJECT_ROOT, "assets")

usd_files = {
    "milk_carton": os.path.join(ASSETS_DIR, "milk", "meshes", "milk_carton_scene.usd"),
    "wooden_box": os.path.join(ASSETS_DIR, "wooden_box", "meshes", "box_scene.usd"),
    "milk_model": os.path.join(ASSETS_DIR, "milk", "meshes", "model.usd"),
    "wooden_box_model": os.path.join(ASSETS_DIR, "wooden_box", "meshes", "model.usd"),
}

for name, path in usd_files.items():
    print(f"\n=== {name} ===", flush=True)
    if not os.path.exists(path):
        print("  DOES NOT EXIST", flush=True)
        continue

    stage = Usd.Stage.Open(path)
    root = stage.GetDefaultPrim() or stage.GetPseudoRoot()

    all_meshes = []
    physics_schemas = []

    def walk(prim, depth=0):
        applied = [str(s) for s in prim.GetAppliedSchemas()]
        for s in applied:
            if any(k in s for k in ["RigidBody", "Collision", "Mass"]):
                physics_schemas.append((str(prim.GetPath()), s))
        if UsdGeom.Mesh(prim):
            mesh = UsdGeom.Mesh(prim)
            pts = mesh.GetPointsAttr().Get()
            if pts:
                pts_list = list(pts)
                min_pt = [min(p[i] for p in pts_list) for i in range(3)]
                max_pt = [max(p[i] for p in pts_list) for i in range(3)]
                ext = [max_pt[i] - min_pt[i] for i in range(3)]
                purpose = prim.GetAttribute("purpose").Get() or "default"
                all_meshes.append((str(prim.GetPath()), len(pts_list), ext, min_pt, purpose))
        for child in prim.GetChildren():
            walk(child, depth + 1)

    walk(root)

    print(f"  Root prim: {root.GetPath()}  type={root.GetTypeName()}", flush=True)
    print(f"  Physics schemas found: {len(physics_schemas)}", flush=True)
    for prim_path, schema in physics_schemas:
        print(f"    {prim_path} -> {schema}", flush=True)

    print(f"  Meshes found: {len(all_meshes)}", flush=True)
    if all_meshes:
        all_min = [min(m[3][i] for m in all_meshes) for i in range(3)]
        all_max = [max(m[3][i] + m[2][i] for m in all_meshes) for i in range(3)]
        global_extent = [all_max[i] - all_min[i] for i in range(3)]
        print(f"  Combined extent: [{global_extent[0]:.2f}, {global_extent[1]:.2f}, {global_extent[2]:.2f}]", flush=True)
        print(f"  Combined bbox: min=[{all_min[0]:.2f}, {all_min[1]:.2f}, {all_min[2]:.2f}] max=[{all_max[0]:.2f}, {all_max[1]:.2f}, {all_max[2]:.2f}]", flush=True)
        for mpath, nverts, ext, min_pt, purpose in all_meshes[:3]:
            print(f"    {mpath}: {nverts}v, extent={[f'{e:.2f}' for e in ext]}, purpose={purpose}", flush=True)
        if len(all_meshes) > 3:
            print(f"    ... +{len(all_meshes)-3} more meshes", flush=True)

    # check xform scale on root-level children
    if root.IsValid():
        xf = UsdGeom.Xformable(root)
        ops = xf.GetOrderedXformOps()
        if ops:
            for op in ops:
                print(f"    Xform op: {op.GetName()}", flush=True)
        for child in root.GetChildren():
            cxf = UsdGeom.Xformable(child)
            cops = cxf.GetOrderedXformOps()
            if cops:
                for op in cops:
                    print(f"    Child {child.GetPath()} xform op: {op.GetName()}", flush=True)

print("\nDone.", flush=True)
simulation_app.close()
