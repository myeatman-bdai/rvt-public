import os

import numpy as np
import open3d as o3d
from PIL import Image
import torch

from clip import tokenize

import rvt.mvt.config as default_mvt_cfg
import rvt.config as default_exp_cfg
import rvt.models.rvt_agent as rvt_agent
from rvt.mvt.mvt import MVT
from rvt.utils.rvt_utils import load_agent as load_agent_state
from rvt.utils.peract_utils import IMAGE_SIZE, SCENE_BOUNDS, CAMERAS
from rvt.libs.peract.helpers import utils

MODEL_FOLDER = "runs/rvt"
MODEL_PATH = os.path.join(MODEL_FOLDER, "model_14.pth")
EVAL_LOG_DIR = "/tmp/eval"
EVAL_DATAFOLDER = "data/test"  # TODO
EPISODE_LENGTH = 25  # TODO
EVAL_EPISODES = 25  # TODO


def transform_pcd(pcd, T):

    pcd = np.concatenate([pcd, np.ones_like(pcd)[..., 0: 1]], axis=-1)
    pcd = pcd @ T.T
    pcd = pcd[..., :-1]
    return pcd


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
        cameras=["camera_0", "camera_1"],
        log_dir=f"{EVAL_LOG_DIR}/eval_run",
        **exp_cfg.peract,
        **exp_cfg.rvt,
    )

    agent.build(training=False, device=torch.device(device))
    load_agent_state(MODEL_PATH, agent)
    agent.eval()
    agent.load_clip()
    return agent


def main():

    agent = setup_agent()

    low_dim_state = np.array([1., 0.04, 0.04, 1.])

    img_1 = np.array(Image.open("0_color.png")).transpose((2, 0, 1))
    img_2 = np.array(Image.open("1_color.png")).transpose((2, 0, 1))

    pcd_1 = np.load("0_points.npy").reshape((720, 1280, 3))
    pcd_2 = np.load("1_points.npy").reshape((720, 1280, 3))

    T = np.array([
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 1.2],
        [0., 0., 0., 1.]
    ])
    pcd_1 = transform_pcd(pcd_1, T)
    pcd_2 = transform_pcd(pcd_2, T)

    pcd_1 = pcd_1.transpose((2, 0, 1))
    pcd_2 = pcd_2.transpose((2, 0, 1))

    obs = {
        "camera_0_rgb": img_1,
        "camera_1_rgb": img_2,
        "camera_0_point_cloud": pcd_1.astype(np.float32),
        "camera_1_point_cloud": pcd_2.astype(np.float32),
        "low_dim_state": low_dim_state.astype(np.float32),
        "lang_goal_tokens": tokenize("pick block")[0],
    }

    print()
    for key in obs.keys():
        print(key, obs[key].shape, obs[key].dtype)

    timesteps = 1 
    obs_history = {k: [np.array(v)] * timesteps for k, v in obs.items()}
    prepped_data = {k:torch.tensor(np.array(v)[None], device="cuda:0") for k, v in obs_history.items()}

    output = agent.act(0, prepped_data)
    action = output.action

    x, y, z = action[:3]
    q0, q1, q2, q3 = action[3: 3 + 4]
    gripper_bit, collision_bit = action[3 + 4: 3 + 4 + 2]

    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(np.concatenate(
        [pcd_1.reshape((3, -1)).transpose((1, 0)), pcd_2.reshape((3, -1)).transpose((1, 0))]))
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[x, y, z])
    o3d.visualization.draw_geometries([o3d_pcd, mesh_frame])

    print(action.shape)


main()
