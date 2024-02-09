import os

import numpy as np
import open3d as o3d
from PIL import Image
import torch

from rlbench.backend import task as rlbench_task
from rlbench.backend.utils import task_file_to_task_class
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.action_mode import MoveArmThenGripper

import rvt.mvt.config as default_mvt_cfg
import rvt.config as default_exp_cfg
import rvt.models.rvt_agent as rvt_agent
from rvt.mvt.mvt import MVT
from rvt.utils.rvt_utils import RLBENCH_TASKS
from rvt.utils.rvt_utils import load_agent as load_agent_state
from rvt.utils.peract_utils import IMAGE_SIZE, SCENE_BOUNDS, CAMERAS
from rvt.utils.custom_rlbench_env import (
    CustomMultiTaskRLBenchEnv2 as CustomMultiTaskRLBenchEnv,
)
from rvt.utils.rlbench_planning import (
    EndEffectorPoseViaPlanning2 as EndEffectorPoseViaPlanning,
)
from rvt.libs.peract.helpers import utils

from clip import tokenize

MODEL_FOLDER = "runs/rvt"
MODEL_PATH = os.path.join(MODEL_FOLDER, "model_14.pth")
EVAL_LOG_DIR = "/tmp/eval"
EVAL_DATAFOLDER = "data/test"  # TODO
EPISODE_LENGTH = 25  # TODO
EVAL_EPISODES = 25  # TODO


def setup_agent():

    # agent
    device = "cuda:0"

    exp_cfg = default_exp_cfg.get_cfg_defaults()
    exp_cfg.merge_from_file(os.path.join(MODEL_FOLDER, "exp_cfg.yaml"))

    mvt_cfg = default_mvt_cfg.get_cfg_defaults()
    mvt_cfg.merge_from_file(os.path.join(MODEL_FOLDER, "mvt_cfg.yaml"))

    mvt_cfg.freeze()

    rvt = MVT(
        renderer_device=device,
        **mvt_cfg,
    )

    agent = rvt_agent.RVTAgent(
        network=rvt.to(device),
        image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
        add_lang=mvt_cfg.add_lang,
        scene_bounds=SCENE_BOUNDS,
        cameras=CAMERAS,
        log_dir=f"{EVAL_LOG_DIR}/eval_run",
        **exp_cfg.peract,
        **exp_cfg.rvt,
    )

    agent.build(training=False, device=torch.device(device))
    load_agent_state(MODEL_PATH, agent)
    agent.eval()
    agent.load_clip()
    return agent


def setup_env():

    # env
    camera_resolution = [IMAGE_SIZE, IMAGE_SIZE]
    obs_config = utils.create_obs_config(CAMERAS, camera_resolution, method_name="")

    gripper_mode = Discrete()
    arm_action_mode = EndEffectorPoseViaPlanning()
    action_mode = MoveArmThenGripper(arm_action_mode, gripper_mode)

    task_files = [
        t.replace(".py", "")
        for t in os.listdir(rlbench_task.TASKS_PATH)
        if t != "__init__.py" and t.endswith(".py")
    ]

    tasks = RLBENCH_TASKS
    task_classes = []
    for task in tasks:
        if task not in task_files:
            raise ValueError("Task %s not recognised!." % task)
        task_classes.append(task_file_to_task_class(task))

    eval_env = CustomMultiTaskRLBenchEnv(
        task_classes=task_classes,
        observation_config=obs_config,
        action_mode=action_mode,
        dataset_root=EVAL_DATAFOLDER,
        episode_length=EPISODE_LENGTH,
        headless=True,
        swap_task_every=EVAL_EPISODES,
        include_lang_goal_in_obs=True,
        time_in_state=True,
        record_every_n=-1,
    )

    eval_env.eval = True
    eval_env.launch()
    return eval_env


def main():

    agent = setup_agent()
    eval_env = setup_env()

    obs = eval_env.reset()
    for key in obs.keys():
        print(key, obs[key].shape, obs[key].dtype)

    print(obs["low_dim_state"])
    print(obs["lang_goal_tokens"])

    # pcds = [obs["left_shoulder_point_cloud"], obs["right_shoulder_point_cloud"], obs["wrist_point_cloud"], obs["front_point_cloud"]]
    # pcds = [pcd.transpose((1, 2, 0)).reshape((-1, 3)) for pcd in pcds]
    # pcds = np.concatenate(pcds, axis=0)

    # np.save("rlbench_lowd.npy", obs["low_dim_state"])
    # np.save("rlbench_pcd.npy", pcds)

    # # $$
    # low_dim_state = np.array([1., 0.04, 0.04, 1.])

    # img_1 = np.array(Image.open("0_color.png")).transpose((2, 0, 1))
    # img_2 = np.array(Image.open("1_color.png")).transpose((2, 0, 1))

    # pcd_1 = np.load("0_points.npy").reshape((720, 1280, 3)).transpose((2, 0, 1))
    # pcd_2 = np.load("1_points.npy").reshape((720, 1280, 3)).transpose((2, 0, 1))

    # obs = {
    #     "camera_0_rgb": img_1,
    #     "camera_1_rgb": img_2,
    #     "camera_0_point_cloud": pcd_1,
    #     "camera_1_point_cloud": pcd_2,
    #     "low_dim_state": low_dim_state,
    #     "lang_goal_tokens": tokenize("pick block")[0],
    # }

    # print()
    # for key in obs.keys():
    #     print(key, obs[key].shape)
    # # $$

    # print(pcds.shape)
    # o3d_pcd = o3d.geometry.PointCloud()
    # o3d_pcd.points = o3d.utility.Vector3dVector(pcds)
    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([o3d_pcd, mesh_frame])

    timesteps = 1 
    obs_history = {k: [np.array(v)] * timesteps for k, v in obs.items()}
    prepped_data = {k:torch.tensor(np.array(v)[None], device="cuda:0") for k, v in obs_history.items()}

    y = agent.act(0, prepped_data)
    print(y.action)


main()
