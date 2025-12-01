
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import sys

# Mock isaacsim modules before importing controllers
sys.modules["isaacsim"] = MagicMock()
sys.modules["isaacsim.core"] = MagicMock()
sys.modules["isaacsim.core.utils"] = MagicMock()
sys.modules["isaacsim.core.utils.rotations"] = MagicMock()
sys.modules["isaacsim.core.utils.types"] = MagicMock()
sys.modules["isaacsim.core.prims"] = MagicMock()
sys.modules["isaacsim.robot"] = MagicMock()
sys.modules["isaacsim.robot.manipulators"] = MagicMock()
sys.modules["isaacsim.robot.manipulators.grippers"] = MagicMock()
sys.modules["isaacsim.robot.manipulators.grippers.parallel_gripper"] = MagicMock()
sys.modules["isaacsim.robot.manipulators.controllers"] = MagicMock()
sys.modules["isaacsim.robot.manipulators.examples"] = MagicMock()
sys.modules["isaacsim.robot.manipulators.examples.universal_robots"] = MagicMock()
sys.modules["isaacsim.robot.manipulators.examples.universal_robots.controllers"] = MagicMock()
# Mock the leaf module specifically so "from ... import ..." works
rmp_module = MagicMock()
sys.modules["isaacsim.robot.manipulators.examples.universal_robots.controllers.rmpflow_controller"] = rmp_module
sys.modules["isaacsim.robot_motion"] = MagicMock()
sys.modules["isaacsim.robot_motion.motion_generation"] = MagicMock()
sys.modules["carb"] = MagicMock()

# Mock ArticulationAction specifically to have joint_positions attribute
class MockArticulationAction:
    def __init__(self, joint_positions=None):
        self.joint_positions = joint_positions

sys.modules["isaacsim.core.utils.types"].ArticulationAction = MockArticulationAction

# Explicitly mock classes imported from prims
prims_mock = MagicMock()
prims_mock.Articulation = MagicMock
prims_mock.SingleArticulation = MagicMock
sys.modules["isaacsim.core.prims"] = prims_mock

sys.modules["isaacsim.core.prims"] = prims_mock

# Mocks for PushManipulationController
sys.modules["isaacsim.core.api"] = MagicMock()
sys.modules["isaacsim.core.api.controllers"] = MagicMock()
sys.modules["isaacsim.core.api.controllers.base_controller"] = MagicMock()
sys.modules["isaacsim.core.utils.stage"] = MagicMock()
sys.modules["isaacsim.robot.manipulators.grippers.gripper"] = MagicMock()
sys.modules["isaacsim.robot_motion.motion_generation.articulation_kinematics_solver"] = MagicMock()

# Now import the controllers
# We need to make sure the imports in the controller files work.
# The controller files import from utils_robots.controllers... which might be tricky if not in path.
# We will assume the test is run from the root or we add the path.
sys.path.append("/home/vlad/IsaacLab/vlad/master_isaac/Vlad-AILAB")

# Mock the base classes
class MockPickPlaceController:
    def __init__(self, *args, **kwargs):
        pass
    def reset(self):
        pass

class MockMotionPolicyController:
    def __init__(self, name, articulation_motion_policy, **kwargs):
        self._motion_policy = MagicMock() # Used in UR5eRMPFlowController
        self._articulation_motion_policy = articulation_motion_policy
        # Ensure _robot_articulation exists and has get_world_pose
        if not hasattr(self._articulation_motion_policy, "_robot_articulation"):
             self._articulation_motion_policy._robot_articulation = MagicMock()
             self._articulation_motion_policy._robot_articulation.get_world_pose.return_value = (np.zeros(3), np.zeros(4))

    def reset(self):
        pass

class MockArticulationMotionPolicy:
    def __init__(self, *args, **kwargs):
        self._robot_articulation = MagicMock()
        self._robot_articulation.get_world_pose.return_value = (np.zeros(3), np.zeros(4))

import isaacsim.robot.manipulators.controllers as manipulators_controllers
manipulators_controllers.PickPlaceController = MockPickPlaceController
import isaacsim.robot_motion.motion_generation as mg
mg.MotionPolicyController = MockMotionPolicyController
mg.ArticulationMotionPolicy = MockArticulationMotionPolicy

# Import the files to test
# We need to load the source files directly or import them if they are in the python path.
# Since we are in a specific directory structure, let's try to import them.
# But first we need to patch the imports INSIDE the files if they are not standard.
# The files import:
# from utils_robots.controllers.RMPFflow_pickplace import RMPFlowController (in push_manipulation_controller.py)
# This seems to be a circular or local import. We should mock it.
sys.modules["utils_robots"] = MagicMock()
sys.modules["utils_robots.controllers"] = MagicMock()
sys.modules["utils_robots.controllers.RMPFflow_pickplace"] = MagicMock()
# Also mock the module that pick_place_controller_rmpflow.py might be imported as, to satisfy push_manipulation_controller.py
pick_place_module = MagicMock()
# We need to attach UR5eRMPFlowController to this mock because push_manipulation_controller imports it
pick_place_module.UR5eRMPFlowController = MagicMock()
sys.modules["utils_robots.controllers.pick_place_controller_rmpflow"] = pick_place_module

# Now we can try to import the classes. 
# We will use exec to load the classes from the files because of the complex environment.

def load_class_from_file(filepath, class_name):
    with open(filepath, 'r') as f:
        code = f.read()
    # We need to execute in a context where mocks are available
    # Use a single scope to simulate a module namespace
    module_scope = {}
    # Copy globals to module scope so it has access to mocked modules if needed (though sys.modules handles imports)
    # Actually, imports look at sys.modules, so we don't need to copy globals.
    # But we might need __builtins__.
    module_scope["__builtins__"] = __builtins__
    
    exec(code, module_scope, module_scope)
    return module_scope[class_name]

PickPlaceController = load_class_from_file(
    "/home/vlad/IsaacLab/vlad/master_isaac/Vlad-AILAB/utils_robots/controllers/pick_place_controller_rmpflow.py",
    "RMPFlowPickPlaceController"
)

PushController = load_class_from_file(
    "/home/vlad/IsaacLab/vlad/master_isaac/Vlad-AILAB/utils_robots/controllers/push_manipulation_controller.py",
    "PushManipulationController"
)

class TestSafetyChecks(unittest.TestCase):
    def test_pick_place_safety(self):
        controller = PickPlaceController(
            name="test_pick",
            gripper=MagicMock(),
            robot_articulation=MagicMock()
        )
        
        # Test Safe Action
        safe_action = MockArticulationAction(joint_positions=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        try:
            controller._check_action_safety(safe_action)
        except RuntimeError:
            self.fail("_check_action_safety raised RuntimeError unexpectedly for safe action")

        # Test NaN
        nan_action = MockArticulationAction(joint_positions=np.array([0.0, np.nan, 0.0]))
        with self.assertRaises(RuntimeError):
            controller._check_action_safety(nan_action)
            
        # Test Inf
        inf_action = MockArticulationAction(joint_positions=np.array([0.0, np.inf, 0.0]))
        with self.assertRaises(RuntimeError):
            controller._check_action_safety(inf_action)
            
        # Test Extreme Value
        extreme_action = MockArticulationAction(joint_positions=np.array([0.0, 200.0, 0.0]))
        with self.assertRaises(RuntimeError):
            controller._check_action_safety(extreme_action)

        # Test None values (should be safe/ignored)
        none_action = MockArticulationAction(joint_positions=[None, None, None, None, None, None])
        try:
            controller._check_action_safety(none_action)
        except Exception as e:
            self.fail(f"_check_action_safety raised exception for None values: {e}")

    def test_push_safety(self):
        controller = PushController(
            name="test_push",
            gripper=MagicMock(),
            robot_articulation=MagicMock()
        )
        
        # Test Safe Action
        safe_action = MockArticulationAction(joint_positions=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        try:
            controller._check_action_safety(safe_action)
        except RuntimeError:
            self.fail("_check_action_safety raised RuntimeError unexpectedly for safe action")

        # Test NaN
        nan_action = MockArticulationAction(joint_positions=np.array([0.0, np.nan, 0.0]))
        with self.assertRaises(RuntimeError):
            controller._check_action_safety(nan_action)
            
        # Test Extreme Value
        extreme_action = MockArticulationAction(joint_positions=np.array([0.0, 200.0, 0.0]))
        with self.assertRaises(RuntimeError):
            controller._check_action_safety(extreme_action)

if __name__ == '__main__':
    unittest.main()
