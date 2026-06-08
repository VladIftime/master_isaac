#!/usr/bin/env python3
"""Add RigidBodyAPI and physics tags to ping pong USD meshes.

Usage:
    source /home/vladi/IsaacLab/master_isaac/.master_venv/bin/activate
    cd pingpong_dual_arm
    python scripts/add_physics_to_usd.py

This reads the existing visual-only USD files and adds:
  - RigidBodyAPI (required for Isaac Lab RigidObject)
  - CollisionAPI + collision geometry
  - Physics material (restitution, friction)

Outputs:
    assets/pingpong/table.usd   (with physics, from meshes/.../table.usdc)
    assets/pingpong/racket.usd  (with physics, from meshes/.../racket.usdc)
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

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

MESHES_DIR = os.path.join(_PROJECT_ROOT, "meshes", "pingpong")
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "assets", "pingpong")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def add_physics_to_mesh(
    input_usd,
    output_usd,
    mass=None,
    kinematic=False,
    restitution=0.5,
    static_friction=0.5,
    dynamic_friction=0.5,
):
    """Open a visual USD mesh, add RigidBodyAPI + collision + physics, save."""
    print(f"  Input:  {input_usd}")
    print(f"  Output: {output_usd}")

    # Copy the original USD stage
    src_stage = Usd.Stage.Open(input_usd)
    dst_stage = Usd.Stage.CreateNew(output_usd)
    UsdGeom.SetStageUpAxis(dst_stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(dst_stage, 1.0)

    # Flatten-copy all prims from source
    root_prim = dst_stage.GetPseudoRoot()
    for src_prim in src_stage.Traverse():
        if src_prim.IsPseudoRoot():
            continue
        src_path = str(src_prim.GetPath())
        dst_stage.OverridePrim(src_path)
        Sdf.CopySpec(
            src_stage.GetRootLayer(), src_path, dst_stage.GetRootLayer(), src_path
        )

    # Find the first mesh prim as the root
    mesh_root = None
    for prim in dst_stage.Traverse():
        if prim.GetTypeName() == "Xform" and not prim.IsPseudoRoot():
            mesh_root = prim
            break
    if mesh_root is None:
        for prim in dst_stage.Traverse():
            if not prim.IsPseudoRoot():
                mesh_root = prim
                break
    if mesh_root is None:
        print("  WARNING: no non-root prim found, using pseudo-root")
        mesh_root = dst_stage.GetPseudoRoot()

    mesh_root.SetKind("component")

    # Apply RigidBodyAPI
    rb_api = UsdPhysics.RigidBodyAPI.Apply(mesh_root)
    rb_api.CreateKinematicEnabledAttr(kinematic)

    if mass is not None and not kinematic:
        mass_api = UsdPhysics.MassAPI.Apply(mesh_root)
        mass_api.CreateMassAttr(mass)

    # Find all mesh prims and apply CollisionAPI
    coll_count = 0
    for prim in dst_stage.Traverse():
        if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Sphere) or prim.IsA(UsdGeom.Cube):
            PhysxSchema.PhysxCollisionAPI.Apply(prim)
            coll_api = UsdPhysics.CollisionAPI.Apply(prim)
            coll_api.CreateCollisionEnabledAttr(True)
            coll_count += 1

    if coll_count == 0:
        # No mesh prims found — add collision from bounding box
        print("  No mesh prims found, adding box collision")
        bbox = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
        ).ComputeWorldBound(mesh_root)
        bbox_range = bbox.GetRange()
        center = (bbox_range.GetMin() + bbox_range.GetMax()) * 0.5
        size = bbox_range.GetSize()

        collision_prim = UsdGeom.Cube.Define(
            dst_stage, mesh_root.GetPath().AppendChild("collision")
        )
        collision_prim.AddTranslateOp().Set(Gf.Vec3d(center[0], center[1], center[2]))
        collision_prim.AddScaleOp().Set(Gf.Vec3d(size[0] / 2, size[1] / 2, size[2] / 2))
        PhysxSchema.PhysxCollisionAPI.Apply(collision_prim.GetPrim())
        col_api = UsdPhysics.CollisionAPI.Apply(collision_prim.GetPrim())
        col_api.CreateCollisionEnabledAttr(True)

    # Add physics material to the root prim
    mat_path = mesh_root.GetPath().AppendChild("physicsMaterial")
    mat = UsdShade.Material.Define(dst_stage, mat_path)
    phys_mat = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    phys_mat.CreateStaticFrictionAttr(static_friction)
    phys_mat.CreateDynamicFrictionAttr(dynamic_friction)
    phys_mat.CreateRestitutionAttr(restitution)

    # Bind material to all collision prims
    for prim in dst_stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                UsdShade.Material.Get(dst_stage, mat_path)
            )

    dst_stage.SetDefaultPrim(mesh_root)
    dst_stage.GetRootLayer().Save()
    print(f"  Done: {os.path.getsize(output_usd)} bytes")
    return output_usd


# ---- Table ----
table_input = os.path.join(MESHES_DIR, "ping_pong_", "ping_pong_table", "table.usdc")
if os.path.exists(table_input):
    add_physics_to_mesh(
        table_input,
        os.path.join(OUTPUT_DIR, "table.usd"),
        kinematic=True,
        restitution=0.877,
        static_friction=0.0,
        dynamic_friction=0.0,
    )
else:
    print(f"Table USD not found: {table_input}")

# ---- Racket ----
racket_input = os.path.join(MESHES_DIR, "racket.usdc")
if os.path.exists(racket_input):
    add_physics_to_mesh(
        racket_input,
        os.path.join(OUTPUT_DIR, "racket.usd"),
        kinematic=True,
        restitution=1.0,
        static_friction=0.5,
        dynamic_friction=0.5,
    )
else:
    print(f"Racket USD not found: {racket_input}")

print("\nDone. Converted assets saved to:", OUTPUT_DIR)
simulation_app.close()
