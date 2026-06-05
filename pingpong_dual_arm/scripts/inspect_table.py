#!/usr/bin/env python3
import os
from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
app = AppLauncher(parser.parse_args([]))
simulation_app = app.app

from pxr import Usd, UsdGeom

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
usd = os.path.join(
    _PKG, "meshes", "pingpong", "ping_pong_", "ping_pong_table", "table_extra_parts.usd"
)
stage = Usd.Stage.Open(usd)
for prim in stage.Traverse():
    if prim.IsA(UsdGeom.Mesh):
        mesh = UsdGeom.Mesh(prim)
        extent = mesh.GetExtentAttr().Get()
        dx = extent[1][0] - extent[0][0]
        dy = extent[1][1] - extent[0][1]
        dz = extent[1][2] - extent[0][2]
        cx = (extent[0][0] + extent[1][0]) / 2
        cy = (extent[0][1] + extent[1][1]) / 2
        cz = (extent[0][2] + extent[1][2]) / 2
        print(f"{prim.GetPath()}:")
        print(
            f"  extent=({extent[0][0]:.4f},{extent[0][1]:.4f},{extent[0][2]:.4f}) -> ({extent[1][0]:.4f},{extent[1][1]:.4f},{extent[1][2]:.4f})"
        )
        print(f"  size=({dx:.4f}, {dy:.4f}, {dz:.4f})")
        print(f"  center=({cx:.4f}, {cy:.4f}, {cz:.4f})")

simulation_app.close()
