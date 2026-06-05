#!/usr/bin/env python3
"""Convert ping pong STL/DAE meshes to USD for use in Isaac Lab.

Usage:
    ../../isaaclab.sh -p scripts/convert_pingpong_assets.py

Generates:
    assets/pingpong/table.usd
    assets/pingpong/ball.usd
    assets/pingpong/racket.usd
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MESHES_DIR = os.path.join(_PKG_ROOT, "meshes", "pingpong")
_OUTPUT_DIR = os.path.join(_PKG_ROOT, "assets", "pingpong")

os.makedirs(_OUTPUT_DIR, exist_ok=True)

# ---- Ball properties (from SDF) ----
BALL_RADIUS = 0.02
BALL_MASS = 0.0027
BALL_RESTITUTION = 0.95

# ---- Table properties (from SDF) ----
TABLE_THICKNESS = 0.02
TABLE_RESTITUTION = 0.877

# ---- Racket dimensions ----
RACKET_WIDTH = 0.16
RACKET_HEIGHT = 0.16
RACKET_THICKNESS = 0.01


def _add_physics_material(
    stage, prim_path, restitution, static_friction, dynamic_friction
):
    """Add a physics material to the given prim."""
    material_path = prim_path + "/physicsMaterial"
    UsdShade.Material.Define(stage, material_path)
    material = UsdPhysics.MaterialAPI.Define(stage, material_path)
    if material:
        material.CreateStaticFrictionAttr().Set(static_friction)
        material.CreateDynamicFrictionAttr().Set(dynamic_friction)
        material.CreateRestitutionAttr().Set(restitution)
    return material_path


def create_ball_usd():
    """Create a sphere USD for the ping pong ball."""
    output = os.path.join(_OUTPUT_DIR, "ball.usd")
    stage = Usd.Stage.CreateNew(output)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = stage.DefinePrim("/ball", "Xform")
    stage.SetDefaultPrim(root)
    root.SetKind("component")

    # Visual sphere
    visual = UsdGeom.Sphere.Define(stage, "/ball/visuals/visual")
    visual.CreateRadiusAttr(BALL_RADIUS)

    # Collision sphere
    collision = UsdGeom.Sphere.Define(stage, "/ball/collisions/collision")
    collision.CreateRadiusAttr(BALL_RADIUS)
    PhysxSchema.PhysxCollisionAPI.Apply(collision.GetPrim())
    collision_api = UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
    collision_api.CreateCollisionEnabledAttr(True)

    # Physics material
    mat_path = _add_physics_material(
        stage,
        "/ball/collisions/collision",
        restitution=BALL_RESTITUTION,
        static_friction=0.0,
        dynamic_friction=0.0,
    )
    UsdPhysics.MaterialBindingAPI.Apply(collision.GetPrim()).Bind(
        UsdShade.Material.Get(stage, mat_path)
    )

    # Rigid body
    rb_api = UsdPhysics.RigidBodyAPI.Apply(root)
    rb_api.CreateKinematicEnabledAttr(False)

    # Mass
    mass_api = UsdPhysics.MassAPI.Apply(root)
    mass_api.CreateMassAttr(BALL_MASS)

    stage.GetRootLayer().Save()
    print(f"  Created {output}")
    return output


def create_table_usd():
    """Create a cuboid USD for the ping pong table (placeholder)."""
    output = os.path.join(_OUTPUT_DIR, "table.usd")
    stage = Usd.Stage.CreateNew(output)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = stage.DefinePrim("/table", "Xform")
    stage.SetDefaultPrim(root)
    root.SetKind("component")

    # Visual boxes (table top + legs)
    table_top = UsdGeom.Cube.Define(stage, "/table/visuals/top")
    table_top.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.38))
    table_top.AddScaleOp().Set(Gf.Vec3d(1.525, 2.74, TABLE_THICKNESS))

    # Collision
    collision = UsdGeom.Cube.Define(stage, "/table/collisions/collision")
    collision.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.38))
    collision.AddScaleOp().Set(Gf.Vec3d(1.525, 2.74, TABLE_THICKNESS))

    PhysxSchema.PhysxCollisionAPI.Apply(collision.GetPrim())
    collision_api = UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
    collision_api.CreateCollisionEnabledAttr(True)

    mat_path = _add_physics_material(
        stage,
        "/table/collisions/collision",
        restitution=TABLE_RESTITUTION,
        static_friction=0.0,
        dynamic_friction=0.0,
    )
    UsdPhysics.MaterialBindingAPI.Apply(collision.GetPrim()).Bind(
        UsdShade.Material.Get(stage, mat_path)
    )

    stage.GetRootLayer().Save()
    print(f"  Created {output}")
    return output


def create_racket_usd():
    """Create a cuboid USD for the ping pong racket (placeholder)."""
    output = os.path.join(_OUTPUT_DIR, "racket.usd")
    stage = Usd.Stage.CreateNew(output)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = stage.DefinePrim("/racket", "Xform")
    stage.SetDefaultPrim(root)
    root.SetKind("component")

    # Racket head
    head = UsdGeom.Cylinder.Define(stage, "/racket/visuals/head")
    head.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
    head.AddScaleOp().Set(
        Gf.Vec3d(RACKET_WIDTH / 2, RACKET_HEIGHT / 2, RACKET_THICKNESS)
    )

    # Handle (stick)
    handle = UsdGeom.Cylinder.Define(stage, "/racket/visuals/handle")
    handle.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.06))
    handle.AddScaleOp().Set(Gf.Vec3d(0.015, 0.015, 0.05))

    # Collision
    collision = UsdGeom.Cube.Define(stage, "/racket/collisions/collision")
    collision.AddScaleOp().Set(Gf.Vec3d(RACKET_WIDTH, RACKET_HEIGHT, RACKET_THICKNESS))

    PhysxSchema.PhysxCollisionAPI.Apply(collision.GetPrim())
    collision_api = UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
    collision_api.CreateCollisionEnabledAttr(True)

    mat_path = _add_physics_material(
        stage,
        "/racket/collisions/collision",
        restitution=1.0,
        static_friction=0.5,
        dynamic_friction=0.5,
    )
    UsdPhysics.MaterialBindingAPI.Apply(collision.GetPrim()).Bind(
        UsdShade.Material.Get(stage, mat_path)
    )

    stage.GetRootLayer().Save()
    print(f"  Created {output}")
    return output


if __name__ == "__main__":
    print(f"Converting ping pong assets...")
    print(f"  Meshes dir: {_MESHES_DIR}")
    print(f"  Output dir: {_OUTPUT_DIR}")
    create_ball_usd()
    create_table_usd()
    create_racket_usd()
    print(f"\nDone. Converted assets saved to {_OUTPUT_DIR}/")
