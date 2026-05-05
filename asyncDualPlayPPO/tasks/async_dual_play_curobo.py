"""
Environment configuration for asymmetric dual-play — cuRobo IK variant.

Re-uses the DiffIK scene/events/terminations (JointPositionActionCfg arm).
train_curobo.py overrides the arm action to JointPositionActionCfg at runtime
and drives joints directly from the cuRobo IK solution, so the DiffIK controller
is never actually invoked — the action spec only defines the action-space shape
(7D: 6 arm joints + 1 gripper).

Observations and reward structure are identical to async_dual_play_diffik.py.
"""

from asyncDualPlayPPO.tasks.async_dual_play_diffik import AsyncDualPlayDiffIKEnvCfg  # noqa: F401

# Expose under the cuRobo name so train_curobo.py can import it unambiguously.
AsyncDualPlayCuRoboEnvCfg = AsyncDualPlayDiffIKEnvCfg
