"""
Generate a T-shape USD file with visual and collision meshes.
No rigid body APIs — Isaac Lab's spawner handles physics setup.
"""
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

TOP_W = 0.08; TOP_D = 0.02; STEM_W = 0.02; STEM_D = 0.06; HEIGHT = 0.05
TOP_CY = STEM_D / 2.0 - TOP_D / 2.0
STEM_CY = -STEM_D / 2.0 + TOP_D / 2.0

BOX_VERTS = [
    (-0.5,-0.5,-0.5),(0.5,-0.5,-0.5),(0.5,0.5,-0.5),(-0.5,0.5,-0.5),
    (-0.5,-0.5,0.5),(0.5,-0.5,0.5),(0.5,0.5,0.5),(-0.5,0.5,0.5),
]
BOX_TRIS = []
for f in [(0,1,2,3),(4,5,6,7),(0,1,5,4),(2,3,7,6),(0,3,7,4),(1,2,6,5)]:
    BOX_TRIS.extend([f[0],f[1],f[2], f[0],f[2],f[3]])

def add_box_mesh(stage, path, sx, sy, sz, cx, cy, cz, purpose="default"):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.GetPurposeAttr().Set(purpose)
    verts = [Gf.Vec3f(x*sx+cx, y*sy+cy, z*sz+cz) for x,y,z in BOX_VERTS]
    mesh.CreatePointsAttr(verts)
    mesh.CreateFaceVertexCountsAttr([3]*12)
    mesh.CreateFaceVertexIndicesAttr(BOX_TRIS)
    xs=[v[0] for v in verts]; ys=[v[1] for v in verts]; zs=[v[2] for v in verts]
    mesh.CreateExtentAttr([Gf.Vec3f(min(xs),min(ys),min(zs)),Gf.Vec3f(max(xs),max(ys),max(zs))])
    return mesh

stage = Usd.Stage.CreateInMemory("t_shape.usda")
stage.SetDefaultPrim(stage.DefinePrim("/t_shape_urdf", "Xform"))
stage.SetMetadata("upAxis", "Z")
stage.SetMetadata("metersPerUnit", 1.0)

root = stage.GetPrimAtPath("/t_shape_urdf")
root.SetKind("component")

# ---- Rigid body on the ROOT prim so Isaac Lab's spawner finds it ----
UsdPhysics.RigidBodyAPI.Apply(root)
UsdPhysics.MassAPI.Apply(root)
# Mass and inertia are left unset — the spawner's MassPropertiesCfg(density=300.0)
# handles both, and PhysX computes correct inertia from collision geometry.

# ---- baseLink ----
base_link = UsdGeom.Xform.Define(stage, "/t_shape_urdf/baseLink")

# ---- Collision meshes (under baseLink) ----
col_scope = UsdGeom.Scope.Define(stage, "/t_shape_urdf/baseLink/colliders")
for name, sx, sy, cy in [("top_bar", TOP_W, TOP_D, TOP_CY), ("stem", STEM_W, STEM_D, STEM_CY)]:
    xf = UsdGeom.Xform.Define(stage, f"/t_shape_urdf/baseLink/colliders/{name}")
    xf.AddTranslateOp().Set(Gf.Vec3f(0, cy, HEIGHT/2.0))
    m = add_box_mesh(stage, f"/t_shape_urdf/baseLink/colliders/{name}/mesh", sx, sy, HEIGHT, 0, 0, 0, "guide")
    col = UsdPhysics.CollisionAPI.Apply(m.GetPrim())
    col.GetCollisionEnabledAttr().Set(True)
    mesh_api = UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim())
    mesh_api.GetApproximationAttr().Set("convexHull")

# ---- Visual meshes (same geometry, purpose=default) ----
vis_scope = UsdGeom.Scope.Define(stage, "/t_shape_urdf/baseLink/visuals")
for name, sx, sy, cy in [("top_bar", TOP_W, TOP_D, TOP_CY), ("stem", STEM_W, STEM_D, STEM_CY)]:
    xf = UsdGeom.Xform.Define(stage, f"/t_shape_urdf/baseLink/visuals/{name}")
    xf.AddTranslateOp().Set(Gf.Vec3f(0, cy, HEIGHT/2.0))
    add_box_mesh(stage, f"/t_shape_urdf/baseLink/visuals/{name}/mesh", sx, sy, HEIGHT, 0, 0, 0, "default")

# ---- Material ----
mat = UsdShade.Material.Define(stage, "/t_shape_urdf/Looks/TBlockMat")
shader = UsdShade.Shader.Define(stage, "/t_shape_urdf/Looks/TBlockMat/Shader")
shader.CreateIdAttr("UsdPreviewSurface")
shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.1, 0.8, 0.1))
shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
UsdShade.MaterialBindingAPI(root).Bind(mat)

# ---- Save ----
output_path = "/home/vladi/IsaacLab/master_isaac/asyncDualPlayPPO/assets/blocks/t_shape.usda"
stage.GetRootLayer().Export(output_path)
print(f"[OK] Generated {output_path}")
