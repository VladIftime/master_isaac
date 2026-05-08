import sys

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.types.math import Pose

print("Successfully imported!")
simulation_app.close()
