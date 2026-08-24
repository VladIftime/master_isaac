from isaacsim import SimulationApp
sim = SimulationApp({"headless": True})
import inspect
from isaacsim.robot.manipulators.articulation_kinematics_solver import ArticulationKinematicsSolver
print(inspect.getsource(ArticulationKinematicsSolver))
sim.close()
