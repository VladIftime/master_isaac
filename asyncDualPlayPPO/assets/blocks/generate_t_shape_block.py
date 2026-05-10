"""
Generate T-shaped block USD asset for Push-T validation tests.

Run this inside Isaac Sim:
    ./isaaclab.sh -p asyncDualPlayPPO/assets/blocks/generate_t_shape_block.py

Creates:
    - t_shape.usd              (top-level assembly, binary crate)
    - configuration/t_shape_base.usd
    - configuration/t_shape_physics.usd
    - configuration/t_shape_robot.usd
    - configuration/t_shape_sensor.usd
"""

import os
import sys

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
from isaacsim.core.utils.stage import get_current_stage, open_stage, save_stage

BLOCKS_DIR = os.path.dirname(os.path.abspath(__file__))
CFG_DIR = os.path.join(BLOCKS_DIR, "configuration")

# T-shape dimensions (meters)
TOP_W = 0.08    # top bar width (X) — diffusion_policy 4:1 ratio
TOP_D = 0.02    # top bar depth (Y)
STEM_W = 0.02   # stem width (X)
STEM_D = 0.06   # stem depth (Y) — diffusion_policy 1:3 ratio
HEIGHT = 0.03   # total height (Z)

# Derived positions  so the T is centered at origin
TOP_CY = STEM_D / 2.0 - TOP_D / 2.0               # top bar center Y
STEM_CY = -STEM_D / 2.0 + TOP_D / 2.0              # stem center Y


def create_t_shape_usda():
    """Create t_shape.usda — ASCII USD with visual + collision + physics."""
    usda_path = os.path.join(BLOCKS_DIR, "t_shape.usda")

    stage = Usd.Stage.CreateInMemory("t_shape.usda")
    stage.SetDefaultPrim(stage.DefinePrim("/t_shape", "Xform"))
    stage.SetMetadata("upAxis", "Z")
    stage.SetMetadata("metersPerUnit", 1.0)

    root = stage.GetPrimAtPath("/t_shape")
    root.SetKind("component")
    root.SetTypeName("Xform")

    # ---- Visual meshes (two boxes forming a T) ----
    vis_scope = UsdGeom.Scope.Define(stage, "/t_shape/visuals")

    top_bar = UsdGeom.Cube.Define(stage, "/t_shape/visuals/top_bar")
    top_bar.GetSizeAttr().Set(1.0)
    UsdGeom.Xformable(top_bar).AddScaleOp().Set(Gf.Vec3f(TOP_W, TOP_D, HEIGHT))
    UsdGeom.Xformable(top_bar).AddTranslateOp().Set(Gf.Vec3f(0.0, TOP_CY, HEIGHT / 2.0))
    top_bar.CreatePurposeAttr("default")

    stem = UsdGeom.Cube.Define(stage, "/t_shape/visuals/stem")
    stem.GetSizeAttr().Set(1.0)
    UsdGeom.Xformable(stem).AddScaleOp().Set(Gf.Vec3f(STEM_W, STEM_D, HEIGHT))
    UsdGeom.Xformable(stem).AddTranslateOp().Set(Gf.Vec3f(0.0, STEM_CY, HEIGHT / 2.0))
    stem.CreatePurposeAttr("default")

    # ---- Collision meshes (same two boxes) ----
    col_scope = UsdGeom.Scope.Define(stage, "/t_shape/colliders")

    col_top = UsdGeom.Cube.Define(stage, "/t_shape/colliders/top_bar")
    col_top.GetSizeAttr().Set(1.0)
    UsdGeom.Xformable(col_top).AddScaleOp().Set(Gf.Vec3f(TOP_W, TOP_D, HEIGHT))
    UsdGeom.Xformable(col_top).AddTranslateOp().Set(Gf.Vec3f(0.0, TOP_CY, HEIGHT / 2.0))
    col_top.CreatePurposeAttr("guide")

    col_stem = UsdGeom.Cube.Define(stage, "/t_shape/colliders/stem")
    col_stem.GetSizeAttr().Set(1.0)
    UsdGeom.Xformable(col_stem).AddScaleOp().Set(Gf.Vec3f(STEM_W, STEM_D, HEIGHT))
    UsdGeom.Xformable(col_stem).AddTranslateOp().Set(Gf.Vec3f(0.0, STEM_CY, HEIGHT / 2.0))
    col_stem.CreatePurposeAttr("guide")

    # ---- Physics ----
    UsdPhysics.RigidBodyAPI.Apply(root)
    PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root)

    mass_attr = root.GetAttribute("physics:mass")
    mass_attr.Set(0.1)

    inertia_attr = root.GetAttribute("physics:diagonalInertia")
    if not inertia_attr.IsValid():
        root.CreateAttribute("physics:diagonalInertia", Sdf.ValueTypeNames.Float3)
    root.GetAttribute("physics:diagonalInertia").Set(Gf.Vec3f(1.0, 1.0, 1.0))

    # Collision group
    coll_group_path = "/t_shape/colliders/<collision_group>"
    coll_group = UsdPhysics.CollisionGroup.Define(stage, coll_group_path)
    coll_group.GetFilteredGroupsAttr().Set([])
    coll_group.GetCollisionEnabledAttr().Set(True)

    # Apply collision API and collision group to both collision cubes
    for prim_name in ["top_bar", "stem"]:
        prim = stage.GetPrimAtPath(f"/t_shape/colliders/{prim_name}")
        UsdPhysics.CollisionAPI.Apply(prim)
        # Add to collision group
        rel = prim.CreateRelationship("physics:collisionGroup")
        rel.AddTarget(coll_group.GetPath())

    # ---- Material (bright red for T-block) ----
    mat = UsdShade.Material.Define(stage, "/t_shape/Looks/TBlockMat")
    shader = UsdShade.Shader.Define(stage, "/t_shape/Looks/TBlockMat/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.9, 0.15, 0.15))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    # Bind material to root prim so it applies to all children
    UsdShade.MaterialBindingAPI(root).Bind(mat)

    # ---- Save as ASCII ----
    stage.GetRootLayer().Export(usda_path)
    print(f"[OK] Created {usda_path}")


def main():
    create_t_shape_usda()

    print("\nDone. To convert to binary USD crate (needed by Isaac Lab):")
    print(f"  cd {BLOCKS_DIR}")
    print("  python3 -c \"from pxr import Usd; stage = Usd.Stage.Open('t_shape.usda'); stage.Export('t_shape.usd')\"")
    print("\nThen also create the configuration files (optional, for full compatibility):")
    print("  Or just reference t_shape.usd directly in your test code.")


if __name__ == "__main__":
    main()
