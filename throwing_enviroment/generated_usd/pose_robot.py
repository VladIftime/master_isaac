import json, omni.timeline, omni.usd
from pxr import UsdPhysics, Sdf
joints = json.loads('''{
  "left_shoulder_pan_joint": 0.0,
  "right_shoulder_pan_joint": 7.473566256521735e-06,
  "left_shoulder_lift_joint": -1.5700000524520874,
  "right_shoulder_lift_joint": -1.569998860359192,
  "left_elbow_joint": -1.5700000524520874,
  "right_elbow_joint": 1.570000410079956,
  "left_wrist_1_joint": -1.5700000524520874,
  "right_wrist_1_joint": -1.5699996948242188,
  "left_wrist_2_joint": 1.5700000524520874,
  "right_wrist_2_joint": -1.5698809623718262,
  "left_wrist_3_joint": 0.0,
  "right_wrist_3_joint": -1.5681796412536642e-06,
  "lgripper_finger_joint": 0.0,
  "lgripper_left_inner_knuckle_joint": 0.0,
  "lgripper_right_inner_knuckle_joint": 0.0,
  "lgripper_right_outer_knuckle_joint": 0.0,
  "rgripper_finger_joint": 0.6168190836906433,
  "rgripper_left_inner_knuckle_joint": 0.00015960731252562255,
  "rgripper_right_inner_knuckle_joint": -0.0001582121039973572,
  "rgripper_right_outer_knuckle_joint": -0.0001402247289661318,
  "lgripper_left_inner_finger_joint": 0.0,
  "lgripper_right_inner_finger_joint": 0.0,
  "rgripper_left_inner_finger_joint": -1.5684626930578816e-07,
  "rgripper_right_inner_finger_joint": -3.6735949834110215e-05
}''')
stage = omni.usd.get_context().get_stage()
count = 0
# Debug: print first few prim paths
all_paths = [str(p.GetPath()) for p in stage.TraverseAll()]
print(f'Stage has {len(all_paths)} prims')
for p in all_paths[:10]:
    print(f'  {p}')
for prim in stage.TraverseAll():
    path = str(prim.GetPath())
    for jname, angle in joints.items():
        if jname in path:
            attr = prim.GetAttribute('drive:angular:physics:targetPosition')
            if not attr or not attr.IsValid():
                attr = prim.CreateAttribute('drive:angular:physics:targetPosition', Sdf.ValueTypeNames.Float)
            attr.Set(float(angle))
            count += 1
            print(f'Set {jname} = {angle:.3f}')
            break
print(f'Set {count} joints')
# Auto-play simulation so joints settle
tl = omni.timeline.get_timeline_interface()
tl.play()
