# Dual_arm Isaac Gym Manipulation Benchmark Environments


### About this repository

This repository contains example Dual_arm RL environments for the NVIDIA Isaac Gym.


### Installation

Download the Isaac Gym Preview 4 release from the [website](https://developer.nvidia.com/isaac-gym), then
follow the installation instructions in the documentation. We highly recommend using a conda environment 
to simplify setup.

Ensure that Isaac Gym works on your system by running one of the examples from the `python/examples` 
directory, like `joint_monkey.py`. Follow the troubleshooting steps described in the Isaac Gym Preview 4
install instructions if you have any trouble running the samples.

Once Isaac Gym is installed and samples work within your current Python environment, install this repo:

```bash
pip install -e .
```

### Running the benchmarks

Due to challenges in using IKFast within the conda environment employed by IsaacGym, we have opted to connect IKFast with IsaacGym via ROS. Additionally, due to potential orientation mismatch issues, we recommend using the quaternion (quat) format for all orientation transformations. While other methods are available, we strongly advise sticking with quaternions to ensure consistency. However, the choice is ultimately up to you. To begin, you need to start the IKFast ROS node.

```bash
cd DualArm/dualarmisaacgymenvs/
./ros_ikfast.sh
```
The `num_envs` setting should be adjusted based on your graphics processing power. If running 16 environments causes issues, try reducing the number. It's important to note that in our example, observations are based on visual information, and each environment is equipped with a camera, which consumes additional resources. However, this setup allows for more interactive environments. If visual input is not required, you can remove the cameras and potentially run up to 2048 environments. In summary, the optimal `num_envs` setting depends on your specific tasks, preferences, and the capabilities of your computer.

```bash
conda activate rlgpu
python reach_train.py task=ReachDualArm num_envs=16 headless=false
```
### Loading Objects and Robot Models Notification

First, because there are no publicly available URDF files for the UR5e with Robotiq_140 that are fully compatible with the IsaacGym platform, we have created our own URDF file based on our understanding. We have designed the parameters to allow the gripper to close and open to any width according to the attributes of the Robotiq_140. Therefore, we do not recommend making modifications to the URDF or altering the control method for operating the gripper, as this could lead to unexpected behaviors. However, the choice is ultimately yours.

We also use a text file to import object models, allowing you to load various URDF files with different colors without the need to manually design separate color variants. You can also set any reasonable pose for these objects. While it is possible to load objects in other ways, we do not recommend it due to potential redundant coding and unpredictable errors, such as having to calculate and manage maximum body limits and other complexities. Again, the decision is yours.


### Loading trained models

We leverage the [**skrl reinforcement learning library**](https://skrl.readthedocs.io/en/latest/) to develop our learning models. This library facilitates the seamless integration and modification of policy and value networks, making it an ideal choice for our needs. Additionally, skrl's architecture simplifies the process of loading and utilizing trained models, enabling efficient experimentation and optimization in our reinforcement learning workflows.

### Zooming In and Out

If you're working with a single environment, you may find it challenging to zoom in close enough to get a clear view. Here's a suggestion: first, use your mouse scroll wheel to zoom out significantly until you can see the entire map. Then, adjust the view so that it's facing you directly. After that, zoom in gradually until you can see the entire environment. There are also some useful tricks you can discover while navigating the view, so feel free to explore and experiment with the controls to find what works best for you.

### Configuration and command line arguments

We use [Hydra](https://hydra.cc/docs/intro/) to manage the config. Note that this has some 
differences from previous incarnations in older versions of Isaac Gym.
 
Key arguments to the `train.py` script are:

* `task=TASK` - selects which task to use. Any of `AllegroHand`, `AllegroHandDextremeADR`, `AllegroHandDextremeManualDR`, `AllegroKukaLSTM`, `AllegroKukaTwoArmsLSTM`, `Ant`, `Anymal`, `AnymalTerrain`, `BallBalance`, `Cartpole`, `FrankaCabinet`, `Humanoid`, `Ingenuity` `Quadcopter`, `ShadowHand`, `ShadowHandOpenAI_FF`, `ShadowHandOpenAI_LSTM`, and `Trifinger` (these correspond to the config for each environment in the folder `isaacgymenvs/config/task`)
* `train=TRAIN` - selects which training config to use. Will automatically default to the correct config for the environment (ie. `<TASK>PPO`).
* `num_envs=NUM_ENVS` - selects the number of environments to use (overriding the default number of environments set in the task config).
* `seed=SEED` - sets a seed value for randomizations, and overrides the default seed set up in the task config
* `sim_device=SIM_DEVICE_TYPE` - Device used for physics simulation. Set to `cuda:0` (default) to use GPU and to `cpu` for CPU. Follows PyTorch-like device syntax.
* `rl_device=RL_DEVICE` - Which device / ID to use for the RL algorithm. Defaults to `cuda:0`, and also follows PyTorch-like device syntax.
* `graphics_device_id=GRAPHICS_DEVICE_ID` - Which Vulkan graphics device ID to use for rendering. Defaults to 0. **Note** - this may be different from CUDA device ID, and does **not** follow PyTorch-like device syntax.
* `pipeline=PIPELINE` - Which API pipeline to use. Defaults to `gpu`, can also set to `cpu`. When using the `gpu` pipeline, all data stays on the GPU and everything runs as fast as possible. When using the `cpu` pipeline, simulation can run on either CPU or GPU, depending on the `sim_device` setting, but a copy of the data is always made on the CPU at every step.
* `test=TEST`- If set to `True`, only runs inference on the policy and does not do any training.
* `checkpoint=CHECKPOINT_PATH` - Set to path to the checkpoint to load for training or testing.
* `headless=HEADLESS` - Whether to run in headless mode.
* `experiment=EXPERIMENT` - Sets the name of the experiment.
* `max_iterations=MAX_ITERATIONS` - Sets how many iterations to run for. Reasonable defaults are provided for the provided environments.

Hydra also allows setting variables inside config files directly as command line arguments. As an example, to set the discount rate for a rl_games training run, you can use `train.params.config.gamma=0.999`. Similarly, variables in task configs can also be set. For example, `task.env.enableDebugVis=True`.

#### Hydra Notes

Default values for each of these are found in the `isaacgymenvs/config/config.yaml` file.

The way that the `task` and `train` portions of the config works are through the use of config groups. 
You can learn more about how these work [here](https://hydra.cc/docs/tutorials/structured_config/config_groups/)
The actual configs for `task` are in `isaacgymenvs/config/task/<TASK>.yaml` and for train in `isaacgymenvs/config/train/<TASK>PPO.yaml`. 

In some places in the config you will find other variables referenced (for example,
 `num_actors: ${....task.env.numEnvs}`). Each `.` represents going one level up in the config hierarchy.
 This is documented fully [here](https://omegaconf.readthedocs.io/en/latest/usage.html#variable-interpolation).

## Tasks

Source code for tasks can be found in `isaacgymenvs/tasks`. 

Each task subclasses the `VecEnv` base class in `isaacgymenvs/base/vec_task.py`.

Refer to [docs/framework.md](docs/framework.md) for how to create your own tasks.

Full details on each of the tasks available can be found in the [RL examples documentation](docs/rl_examples.md).

## Add New Task

After creating all related files, remember to add these contents in DualArm/dualarmisaacgymenvs/tasks/__init__.py

```
from .reach_dual_arm import ReachDualArm
from .new_dual_arm import NewDualArm

# Mappings from strings to environments
isaacgym_task_map = {
    "ReachDualArm": ReachDualArm,
    "NewDualArm": NewDualArm,

}
```
### Modifying the Robot’s Appearance

To easily change the robot's clothing, we have created a dedicated folder:

📂 `DualArm/assets/body/`

Inside this folder, there are **two clothing options**:
- **IRL**
- **Intuition**

#### Switching the Robot’s Appearance

To change the robot's appearance, update the file:  
📄 **`DualArm/dualarmisaacgymenvs/tasks/reach_dual_arm.py`**

##### 🏷️ Set the Robot to **IRL**
To use the **IRL** appearance, ensure the following lines are active in `reach_dual_arm.py`:

```python
def add_object_from_file(self, file_name):

    # Define the extra lines to add at the end
    robot_bodies = [
        "body/head.urdf 1.0 1.0 1.0 0.0 -0.26 0.806 0.0 0.77 1.57",
        "body/body_front.urdf 1.0 1.0 1.0 0.0 -0.38 0.0 3.14 0.0 0.0",
        # "body/body_intuition.urdf 1.0 1.0 1.0 0.0 -0.38 0.0 3.14 0.0 0.0",
    ]
```
##### 🏷️ Set the Robot to **Intuition**
To use the **Intuition** appearance, ensure the following lines are active in `reach_dual_arm.py`:

```python
def add_object_from_file(self, file_name):

    # Define the extra lines to add at the end
    robot_bodies = [
        "body/head.urdf 1.0 1.0 1.0 0.0 -0.26 0.806 0.0 0.77 1.57",
        # "body/body_front.urdf 1.0 1.0 1.0 0.0 -0.38 0.0 3.14 0.0 0.0",
        "body/body_intuition.urdf 1.0 1.0 1.0 0.0 -0.38 0.0 3.14 0.0 0.0",
    ]
```
## Domain Randomization

IsaacGymEnvs includes a framework for Domain Randomization to improve Sim-to-Real transfer of trained
RL policies. You can read more about it [here](docs/domain_randomization.md).

## Reproducibility and Determinism

If deterministic training of RL policies is important for your work, you may wish to review our [Reproducibility and Determinism Documentation](docs/reproducibility.md).

## Multi-GPU Training

You can run multi-GPU training using `torchrun` (i.e., `torch.distributed`) using this repository.

Here is an example command for how to run in this way -
`torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py multi_gpu=True task=Ant <OTHER_ARGS>`

Where the `--nproc_per_node=` flag specifies how many processes to run and note the `multi_gpu=True` flag must be set on the train script in order for multi-GPU training to run.

## Population Based Training

You can run population based training to help find good hyperparameters or to train on very difficult environments which would otherwise
be hard to learn anything on without it. See [the readme](docs/pbt.md) for details.

## WandB support

You can run [WandB](https://wandb.ai/) with Isaac Gym Envs by setting `wandb_activate=True` flag from the command line. You can set the group, name, entity, and project for the run by setting the `wandb_group`, `wandb_name`, `wandb_entity` and `wandb_project` set. Make sure you have WandB installed with `pip install wandb` before activating.


## Capture videos


We implement the standard `env.render(mode='rgb_rray')` `gym` API to provide an image of the simulator viewer. Additionally, we can leverage `gym.wrappers.RecordVideo` to help record videos that shows agent's gameplay. Consider running the following file which should produce a video in the `videos` folder.

```python
import gym
import isaacgym
import isaacgymenvs
import torch

num_envs = 64

envs = isaacgymenvs.make(
	seed=0, 
	task="Ant", 
	num_envs=num_envs, 
	sim_device="cuda:0",
	rl_device="cuda:0",
	graphics_device_id=0,
	headless=False,
	multi_gpu=False,
	virtual_screen_capture=True,
	force_render=False,
)
envs.is_vector_env = True
envs = gym.wrappers.RecordVideo(
	envs,
	"./videos",
	step_trigger=lambda step: step % 10000 == 0, # record the videos every 10000 steps
	video_length=100  # for each video record up to 100 steps
)
envs.reset()
print("the image of Isaac Gym viewer is an array of shape", envs.render(mode="rgb_array").shape)
for _ in range(100):
	actions = 2.0 * torch.rand((num_envs,) + envs.action_space.shape, device = 'cuda:0') - 1.0
	envs.step(actions)
```

## Capture videos during training

You can automatically capture the videos of the agents gameplay by toggling the `capture_video=True` flag and tune the capture frequency `capture_video_freq=1500` and video length via `capture_video_len=100`. You can set `force_render=False` to disable rendering when the videos are not captured.

```
python train.py capture_video=True capture_video_freq=1500 capture_video_len=100 force_render=False
```

You can also automatically upload the videos to Weights and Biases:

```
python train.py task=Ant wandb_activate=True wandb_entity=nvidia wandb_project=rl_games capture_video=True force_render=False
```

## Pre-commit

We use [pre-commit](https://pre-commit.com/) to helps us automate short tasks that improve code quality. Before making a commit to the repository, please ensure `pre-commit run --all-files` runs without error.


## Troubleshooting

Please review the Isaac Gym installation instructions first if you run into any issues.

You can either submit issues through GitHub or through the [Isaac Gym forum here](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/isaac-gym/322).

## Citing

Please cite this work as:
```
@misc{makoviychuk2021isaac,
      title={Isaac Gym: High Performance GPU-Based Physics Simulation For Robot Learning}, 
      author={Viktor Makoviychuk and Lukasz Wawrzyniak and Yunrong Guo and Michelle Lu and Kier Storey and Miles Macklin and David Hoeller and Nikita Rudin and Arthur Allshire and Ankur Handa and Gavriel State},
      year={2021},
      journal={arXiv preprint arXiv:2108.10470}
}
```

**Note** if you use the DexPBT: Scaling up Dexterous Manipulation for Hand-Arm Systems with Population Based Training work or the code related to Population Based Training, please cite the following paper:

```
@inproceedings{
	petrenko2023dexpbt,
	author = {Aleksei Petrenko, Arthur Allshire, Gavriel State, Ankur Handa, Viktor Makoviychuk},
	title = {DexPBT: Scaling up Dexterous Manipulation for Hand-Arm Systems with Population Based Training},
	booktitle = {RSS},
	year = {2023}
}
```

**Note** if you use the DeXtreme: Transfer of Agile In-hand Manipulation from Simulation to Reality work or the code related to Automatic Domain Randomisation, please cite the following paper:

```
@inproceedings{
	handa2023dextreme,
	author = {Ankur Handa, Arthur Allshire, Viktor Makoviychuk, Aleksei Petrenko, Ritvik Singh, Jingzhou Liu, Denys Makoviichuk, Karl Van Wyk, Alexander Zhurkevich, Balakumar Sundaralingam, Yashraj Narang, Jean-Francois Lafleche, Dieter Fox, Gavriel State},
	title = {DeXtreme: Transfer of Agile In-hand Manipulation from Simulation to Reality},
	booktitle = {ICRA},
	year = {2023}
} 
```

**Note** if you use the ANYmal rough terrain environment in your work, please ensure you cite the following work:
```
@misc{rudin2021learning,
      title={Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning}, 
      author={Nikita Rudin and David Hoeller and Philipp Reist and Marco Hutter},
      year={2021},
      journal = {arXiv preprint arXiv:2109.11978}
}
```

**Note** if you use the Trifinger environment in your work, please ensure you cite the following work:
```
@misc{isaacgym-trifinger,
  title     = {{Transferring Dexterous Manipulation from GPU Simulation to a Remote Real-World TriFinger}},
  author    = {Allshire, Arthur and Mittal, Mayank and Lodaya, Varun and Makoviychuk, Viktor and Makoviichuk, Denys and Widmaier, Felix and Wuthrich, Manuel and Bauer, Stefan and Handa, Ankur and Garg, Animesh},
  year      = {2021},
  journal = {arXiv preprint arXiv:2108.09779}
}
```

**Note** if you use the AMP: Adversarial Motion Priors environment in your work, please ensure you cite the following work:
```
@article{
	2021-TOG-AMP,
	author = {Peng, Xue Bin and Ma, Ze and Abbeel, Pieter and Levine, Sergey and Kanazawa, Angjoo},
	title = {AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control},
	journal = {ACM Trans. Graph.},
	issue_date = {August 2021},
	volume = {40},
	number = {4},
	month = jul,
	year = {2021},
	articleno = {1},
	numpages = {15},
	url = {http://doi.acm.org/10.1145/3450626.3459670},
	doi = {10.1145/3450626.3459670},
	publisher = {ACM},
	address = {New York, NY, USA},
	keywords = {motion control, physics-based character animation, reinforcement learning},
} 
```

**Note** if you use the Factory simulation methods (e.g., SDF collisions, contact reduction) or Factory learning tools (e.g., assets, environments, or controllers) in your work, please cite the following paper:
```
@inproceedings{
	narang2022factory,
	author = {Yashraj Narang and Kier Storey and Iretiayo Akinola and Miles Macklin and Philipp Reist and Lukasz Wawrzyniak and Yunrong Guo and Adam Moravanszky and Gavriel State and Michelle Lu and Ankur Handa and Dieter Fox},
	title = {Factory: Fast contact for robotic assembly},
	booktitle = {Robotics: Science and Systems},
	year = {2022}
} 
```

**Note** if you use the IndustReal training environments or algorithms in your work, please cite the following paper:
```
@inproceedings{
	tang2023industreal,
	author = {Bingjie Tang and Michael A Lin and Iretiayo Akinola and Ankur Handa and Gaurav S Sukhatme and Fabio Ramos and Dieter Fox and Yashraj Narang},
	title = {IndustReal: Transferring contact-rich assembly tasks from simulation to reality},
	booktitle = {Robotics: Science and Systems},
	year = {2023}
}
```
### COPYRIGHT NOTICE

All rights reserved. This material, including the source code, scripts, and any accompanying documentation, is protected under international copyright laws.

© 2023–Present Yongliang Wang.

This software and related content are intended solely for the authorized use by individuals or organizations who have received prior permission from the copyright holder. No portion of this material may be reproduced, modified, adapted, distributed, transmitted, publicly displayed, or performed in any form or by any means, electronic, mechanical, or otherwise, without the explicit written consent of Yongliang Wang.

Unauthorized use, duplication, or distribution of this material is strictly forbidden and will be prosecuted to the fullest extent of the law, including potential civil liability and criminal penalties under applicable copyright statutes.

For permissions, licensing inquiries, or other information, please contact:
[yongliang.wang@rug.nl]
