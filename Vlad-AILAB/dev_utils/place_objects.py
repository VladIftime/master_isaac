# Get information about YCB Dataset objects
import os
from pathlib import Path
import random
import numpy as np


def get_random_objects(num_objects=1):
    working_dir = os.path.dirname(os.path.realpath(__file__))
    ycb_path = os.path.join(Path(working_dir).parent, "dataset/ycb")
    obj_dirs = [os.path.join(ycb_path, obj_name) for obj_name in os.listdir(ycb_path)]
    obj_dirs.sort()
    object_info = {}
    label2name = {}
    total_object_num = len(obj_dirs)
    for obj_idx, obj_dir in enumerate(obj_dirs):
        usd_file = os.path.join(obj_dir, "final.usd")
        object_info[obj_idx] = {
            "name": os.path.basename(obj_dir),
            "usd_file": usd_file,
            "label": obj_idx,
        }
        label2name[obj_idx] = os.path.basename(obj_dir)

    # Select usd file path for random objects
    objects_list = random.sample(list(object_info.values()), num_objects)
    objects_usd_list = []
    for obj_info in objects_list:
        objects_usd_list.append(obj_info["usd_file"])

    # Print the number and category of randomly generated objects
    for i in range(len(objects_list)):
        print("object_{}: {}".format(i, objects_list[i]["name"]))

    # Specify the positions to generate num_objects objects (if too far, the robot may not reach, if too close, collisions may occur)
    objects_position = np.array([[0.5, 0, 0.1] * num_objects])
    offset = np.array([0, 0, 0.1] * num_objects)

    # Specify the position to place the objects
    target_position = np.array([0.4, 0.4, 0])
    target_orientation = np.array([0, 0, 0, 1])
