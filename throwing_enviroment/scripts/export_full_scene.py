#!/usr/bin/env python3
"""Export the full throwing environment scene as a self-contained USD.

Reads the live simulation's body states (FK result) and bakes the robot
link transforms + joint targets into the exported USD so the robot appears
in the crane pose without needing the Script Editor workflow.

Output:
  generated_usd/full_scene.usd     — reference-based scene (robot, drink, basket, table, stand)
  generated_usd/full_scene_flat.usd — self-contained flattened scene (single file)
  generated_usd/joint_pose.json    — 24 joint angles
  generated_usd/body_poses.json    — body world poses for debugging
  generated_usd/robot_crane_pose/  — modified robot USD with crane-pose link transforms

Usage:
    source /home/vladi/IsaacLab/master_isaac/.master_venv/bin/activate
    cd throwing_enviroment
    python scripts/export_full_scene.py
"""

import glob
import json
import os
import shutil
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
parser.add_argument(
    "--pose",
    type=str,
    default=None,
    help="JSON file with joint position overrides "
    "(e.g. '{\"right_elbow_joint\": 0.0, ...}')",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
app = AppLauncher(args_cli)
simulation_app = app.app

from pxr import Sdf, Usd, UsdGeom, UsdPhysics, Gf
from tasks.throwing_env_cfg import ThrowingEnvCfg
from tasks.throwing_env import ThrowingEnv

OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "generated_usd")
os.makedirs(OUTPUT_DIR, exist_ok=True)
SCENE_OUT = os.path.join(OUTPUT_DIR, "full_scene.usd")
JOINT_OUT = os.path.join(OUTPUT_DIR, "joint_pose.json")
BODY_OUT = os.path.join(OUTPUT_DIR, "body_poses.json")

print("Launching throwing env...", flush=True)
cfg = ThrowingEnvCfg()
cfg.scene.num_envs = 1

if args_cli.pose:
    pose_overrides = (
        json.loads(args_cli.pose)
        if args_cli.pose.startswith("{")
        else (json.load(open(args_cli.pose)) if os.path.isfile(args_cli.pose) else {})
    )
    for jname, jval in pose_overrides.items():
        if jname in cfg.scene.robot.init_state.joint_pos:
            cfg.scene.robot.init_state.joint_pos[jname] = float(jval)
        else:
            print(f"WARNING: joint '{jname}' not in config, ignoring", flush=True)
    print(f"Pose overrides applied: {list(pose_overrides.keys())}", flush=True)

cfg.__post_init__()
env = ThrowingEnv(cfg=cfg)
env.reset()
for _ in range(5):
    env.step(torch.zeros(1, 6, device=env.device))

robot = env.scene["robot"]
milk = env.scene["milk"]
target = env.scene["target"]

root_pos = robot.data.root_pos_w[0]
root_quat = robot.data.root_quat_w[0]

print(
    f"Robot root world pos:  ({root_pos[0]:.3f}, {root_pos[1]:.3f}, {root_pos[2]:.3f})",
    flush=True,
)
print(
    f"Robot root world quat: ({root_quat[0]:.4f}, {root_quat[1]:.4f}, {root_quat[2]:.4f}, {root_quat[3]:.4f})",
    flush=True,
)

joint_map = {}
for i, name in enumerate(robot.joint_names):
    joint_map[name] = float(robot.data.joint_pos[0, i].item())

with open(JOINT_OUT, "w") as f:
    json.dump(joint_map, f, indent=2)
print(f"Joint pose saved: {JOINT_OUT} ({len(joint_map)} joints)", flush=True)

body_world_poses = {}
for i, name in enumerate(robot.body_names):
    pos = robot.data.body_pos_w[0, i].cpu().tolist()
    quat = robot.data.body_quat_w[0, i].cpu().tolist()
    body_world_poses[name] = {"position": pos, "orientation": quat}

with open(BODY_OUT, "w") as f:
    json.dump(body_world_poses, f, indent=2)
print(f"Body poses saved: {BODY_OUT} ({len(body_world_poses)} bodies)", flush=True)

# ── Robot USD cache ────────────────────────────────────────────────
robot_dirs = sorted(
    glob.glob("/tmp/IsaacLab/usd_*/dual_arm_robot.usd"),
    key=os.path.getmtime,
    reverse=True,
)
robot_src = robot_dirs[0] if robot_dirs else None

if robot_src:
    robot_cache_dir = os.path.dirname(robot_src)
    robot_out_dir = os.path.join(OUTPUT_DIR, "robot_crane_pose")
    if os.path.exists(robot_out_dir):
        shutil.rmtree(robot_out_dir)
    shutil.copytree(robot_cache_dir, robot_out_dir)
    robot_ref = os.path.join(robot_out_dir, "dual_arm_robot.usd")
    print(f"Robot USD copied to: {robot_ref}", flush=True)
else:
    robot_ref = None
    print("WARNING: No robot USD found in /tmp/IsaacLab cache", flush=True)

# ── Bake link transforms into the robot USD ───────────────────────
if robot_ref:
    print("Baking link transforms from simulation FK...", flush=True)
    robot_stage = Usd.Stage.Open(robot_ref)

    def _quat_rotate(q, v):
        """Rotate vector v by quaternion q = (w, x, y, z)."""
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        t2 = qw * qx
        t3 = qw * qy
        t4 = qw * qz
        t5 = -qx * qx
        t6 = qx * qy
        t7 = qx * qz
        t8 = -qy * qy
        t9 = qy * qz
        t10 = -qz * qz
        return [
            (2.0 * (t8 + t10) * v[0] + 2.0 * (t6 - t4) * v[1] + 2.0 * (t3 + t7) * v[2])
            + v[0],
            (2.0 * (t4 + t6) * v[0] + 2.0 * (t5 + t10) * v[1] + 2.0 * (t9 - t2) * v[2])
            + v[1],
            (2.0 * (t7 - t3) * v[0] + 2.0 * (t2 + t9) * v[1] + 2.0 * (t5 + t8) * v[2])
            + v[2],
        ]

    def _quat_conjugate(q):
        return [q[0], -q[1], -q[2], -q[3]]

    def _quat_inverse_rotate(q, v):
        return _quat_rotate(_quat_conjugate(q), v)

    def _quat_multiply(a, b):
        return [
            a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
        ]

    rp3 = [root_pos[0].item(), root_pos[1].item(), root_pos[2].item()]
    rq4 = [
        root_quat[0].item(),
        root_quat[1].item(),
        root_quat[2].item(),
        root_quat[3].item(),
    ]
    rq4_inv = _quat_conjugate(rq4)

    updated_links = 0
    for body_name, body_data in body_world_poses.items():
        link_path = f"/ur/{body_name}"
        prim = robot_stage.GetPrimAtPath(link_path)
        if not prim or not prim.IsValid():
            continue

        bp = body_data["position"]
        bq = body_data["orientation"]

        rel_pos = [
            bp[0] - rp3[0],
            bp[1] - rp3[1],
            bp[2] - rp3[2],
        ]
        local_pos = _quat_inverse_rotate(rq4, rel_pos)
        local_quat = _quat_multiply(rq4_inv, bq)

        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        translate_op = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionFloat, "translate")
        translate_op.Set(Gf.Vec3f(*local_pos))
        orient_op = xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat, "orient")
        orient_op.Set(
            Gf.Quatf(local_quat[0], local_quat[1], local_quat[2], local_quat[3])
        )
        scale_op = xf.AddScaleOp(UsdGeom.XformOp.PrecisionFloat, "scale")
        scale_op.Set(Gf.Vec3f(1.0, 1.0, 1.0))
        updated_links += 1

    print(f"  Updated {updated_links} link transforms", flush=True)

    updated_joints = 0
    for prim in robot_stage.TraverseAll():
        path = str(prim.GetPath())
        for jname, angle in joint_map.items():
            if jname in path and prim.IsA(UsdPhysics.RevoluteJoint):
                attr = prim.GetAttribute("drive:angular:physics:targetPosition")
                if not attr or not attr.IsValid():
                    attr = prim.CreateAttribute(
                        "drive:angular:physics:targetPosition",
                        Sdf.ValueTypeNames.Float,
                    )
                attr.Set(float(angle))
                updated_joints += 1
                break

    print(f"  Updated {updated_joints} joint targets", flush=True)
    robot_stage.GetRootLayer().Save()
    print("  Robot USD saved with crane-pose transforms", flush=True)

# ── Build the reference-based scene ───────────────────────────────
print("Building scene...", flush=True)
out_layer = Sdf.Layer.CreateNew(SCENE_OUT)
out_layer.defaultPrim = "World"
Sdf.CreatePrimInLayer(out_layer, "/World")

mp = milk.data.root_pos_w[0]
mq = milk.data.root_quat_w[0]
tp = target.data.root_pos_w[0]
tq = target.data.root_quat_w[0]

entities = {
    "Robot": {
        "type": "ref",
        "ref": robot_ref,
        "pos": (
            root_pos[0].item(),
            root_pos[1].item(),
            root_pos[2].item(),
        ),
        "quat": (
            root_quat[0].item(),
            root_quat[1].item(),
            root_quat[2].item(),
            root_quat[3].item(),
        ),
    },
    "Drink": {
        "type": "ref",
        "ref": os.path.abspath(
            os.path.join(
                _PROJECT_ROOT, "assets", "new_usds", "drink001", "drink_target.usd"
            )
        ),
        "pos": (mp[0].item(), mp[1].item(), mp[2].item()),
        "quat": (mq[0].item(), mq[1].item(), mq[2].item(), mq[3].item()),
    },
    "Target": {
        "type": "ref",
        "ref": os.path.abspath(
            os.path.join(
                _PROJECT_ROOT,
                "assets",
                "new_usds",
                "shopping basket002",
                "basket_target.usd",
            )
        ),
        "pos": (tp[0].item(), tp[1].item(), tp[2].item()),
        "quat": (tq[0].item(), tq[1].item(), tq[2].item(), tq[3].item()),
    },
    "Table": {
        "type": "box",
        "size": (1.0, 1.2, 0.05),
        "pos": (0, 1.0, 0.575),
        "quat": (1, 0, 0, 0),
        "color": (0.6, 0.6, 0.7),
    },
    "Stand": {
        "type": "box",
        "size": (0.5, 0.5, 0.6),
        "pos": (0, 0, 0.3),
        "quat": (1, 0, 0, 0),
        "color": (0.4, 0.4, 0.45),
    },
    "GroundPlane": {
        "type": "plane",
        "pos": (0, 0, 0),
        "color": (0.15, 0.15, 0.15),
    },
}

for name, ent in entities.items():
    prim_path = f"/World/{name}"
    spec = Sdf.CreatePrimInLayer(out_layer, prim_path)
    spec.typeName = "Xform"
    if ent["type"] == "ref" and ent.get("ref"):
        spec.referenceList.prependedItems.append(Sdf.Reference(ent["ref"]))
    elif ent["type"] == "box":
        mesh = Sdf.CreatePrimInLayer(out_layer, f"{prim_path}/{name}Geometry")
        mesh.typeName = "Cube"
    elif ent["type"] == "plane":
        mesh = Sdf.CreatePrimInLayer(out_layer, f"{prim_path}/PlaneGeometry")
        mesh.typeName = "Plane"

out_layer.Save()

stage_out = Usd.Stage.Open(SCENE_OUT)


def _add_material_binding(prim, color):
    path = str(prim.GetPath())
    mat = stage_out.DefinePrim(f"{path}/material", "Material")
    shader = stage_out.DefinePrim(f"{path}/material/shader", "Shader")
    shader.CreateAttribute("info:id", Sdf.ValueTypeNames.Token).Set("UsdPreviewSurface")
    shader.CreateAttribute("inputs:diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*color)
    )
    p = stage_out.GetPrimAtPath(path)
    if p and p.IsValid():
        p.CreateAttribute("material:binding", Sdf.ValueTypeNames.Token).Set(
            f"{path}/material"
        )


for name, ent in entities.items():
    prim = stage_out.GetPrimAtPath(f"/World/{name}")
    if not prim or not prim.IsValid():
        continue
    xf = UsdGeom.Xformable(prim)
    p = ent["pos"]
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionFloat, "translate").Set(Gf.Vec3f(*p))
    if "quat" in ent:
        q = ent["quat"]
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat, "orient").Set(
            Gf.Quatf(q[0], q[1], q[2], q[3])
        )
    if ent["type"] == "box":
        s = ent["size"]
        mesh = stage_out.GetPrimAtPath(f"/World/{name}/{name}Geometry")
        if mesh and mesh.IsValid():
            mxf = UsdGeom.Xformable(mesh)
            s2 = [s[0] / 2, s[1] / 2, s[2] / 2]
            mxf.AddScaleOp(UsdGeom.XformOp.PrecisionFloat, "scale").Set(Gf.Vec3f(*s2))
            _add_material_binding(mesh, ent.get("color", (0.5, 0.5, 0.5)))
    if ent["type"] == "plane":
        mesh = stage_out.GetPrimAtPath(f"/World/{name}/PlaneGeometry")
        if mesh and mesh.IsValid():
            mxf = UsdGeom.Xformable(mesh)
            mxf.AddScaleOp(UsdGeom.XformOp.PrecisionFloat, "scale").Set(
                Gf.Vec3f(3, 3, 1)
            )
            _add_material_binding(mesh, ent.get("color", (0.15, 0.15, 0.15)))

stage_out.GetRootLayer().Save()
print(f"Scene saved: {SCENE_OUT}", flush=True)

# ── Instructions for self-contained file ──────────────────────────
print()
print("=" * 60, flush=True)
print("Export complete.", flush=True)
print(f"  Scene:      {SCENE_OUT}", flush=True)
print(f"  Joint pose: {JOINT_OUT}", flush=True)
print(f"  Body poses: {BODY_OUT}", flush=True)
print()
print("The robot link transforms are baked into the robot USD.", flush=True)
print("Opening full_scene.usd shows the robot in crane pose directly.", flush=True)
print()
print("For a self-contained single-file USD:", flush=True)
print("  1. Open Isaac Sim GUI", flush=True)
print("  2. File > Open > full_scene.usd", flush=True)
print("  3. File > Export > USD (check Flatten)", flush=True)
print("  4. Save as full_scene_flat.usd", flush=True)
print("=" * 60, flush=True)

env.close()
simulation_app.close()
print("Done.", flush=True)
