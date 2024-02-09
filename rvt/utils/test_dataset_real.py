import os
import pickle
import unittest

import clip
import open3d as o3d
import numpy as np

from rvt.utils.dataset_real import extract_obs_real, _get_action, keypoint_discovery_real, _add_keypoints_to_replay_real
from rvt.utils.peract_utils import (
    SCENE_BOUNDS,
    ROTATION_RESOLUTION,
    VOXEL_SIZES,
)

from video_imitation import utils

EPISODE_PATH = "/home/obiza/Code/video_imitation/data/demo_v2_convert/pick_up_can/episode0"


def show_obs_dict(obs_dict, action=None):

    pcd1 = utils.depth_to_pcd(obs_dict["camera1_depth"][0], utils.matrix_to_intrinsics(obs_dict["camera1_intrinsics"]))
    pcd2 = utils.depth_to_pcd(obs_dict["camera2_depth"][0], utils.matrix_to_intrinsics(obs_dict["camera2_intrinsics"]))

    pcd1 = utils.transform_pcd(pcd1, obs_dict["camera1_extrinsics"])
    pcd2 = utils.transform_pcd(pcd2, obs_dict["camera2_extrinsics"])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate([pcd1, pcd2], axis=0))
    pcd.colors = o3d.utility.Vector3dVector(np.concatenate(
        [obs_dict["camera1_rgb"].transpose((1, 2, 0)).reshape((-1, 3)),
         obs_dict["camera2_rgb"].transpose((1, 2, 0)).reshape((-1, 3))], axis=0) / 255.)

    to_show = [pcd]

    if action is not None:
        eef_frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1).transform(action)
        to_show.append(eef_frame)

    o3d.visualization.draw_geometries(to_show)   


class TestDatasetReal(unittest.TestCase):

    def test_extract_obs(self):

        with open(os.path.join(EPISODE_PATH, "metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)

        obs_idx = 5
        t = 0

        obs_dict = extract_obs_real(obs_idx, metadata, t, episode_folder=EPISODE_PATH)

        print("test_extract_obs")
        # show_obs_dict(obs_dict)

    def test_extract_obs_and_action(self):

        with open(os.path.join(EPISODE_PATH, "metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)

        obs_idx = 5
        t = 0

        obs_dict = extract_obs_real(obs_idx, metadata, t, episode_folder=EPISODE_PATH)

        (
            trans_indicies,
            rot_grip_indicies,
            ignore_collisions,
            action,
            attention_coordinates,
        ) =_get_action(obs_idx, obs_idx - 1, metadata, SCENE_BOUNDS, VOXEL_SIZES, ROTATION_RESOLUTION, False)

        pos, quat = action[:3], action[3: 7]
        action = utils.pos_quat_to_transform(pos, quat)

        action2 = utils.pos_quat_to_transform(np.array(attention_coordinates), np.array([1., 0., 0., 0.]))

        print("test_extract_obs_and_action")
        # show_obs_dict(obs_dict, action)
        # show_obs_dict(obs_dict, action2)

    def test_keyframe_discovery(self):

        with open(os.path.join(EPISODE_PATH, "metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)
        
        episode_keypoints = keypoint_discovery_real(metadata)
        print("test_keyframe_discovery")
        print(episode_keypoints)

    def test_add_keypoints_to_replay(self):

        class MockReply:

            def __init__(self):
                self.add_list = []
                self.add_final_list = []
            
            def add(self, *args, **kwargs):
                self.add_list.append((args, kwargs))
            
            def add_final(self, *args, **kwargs):
                self.add_final_list.append((args, kwargs))

        with open(os.path.join(EPISODE_PATH, "metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)

        replay = MockReply()
        task = ""

        clip_model, _ = clip.load("RN50", device="cpu")  # CLIP-ResNet50
        clip_model = clip_model.to("cpu")
        clip_model.eval()

        episode_keypoints = keypoint_discovery_real(metadata)
        _add_keypoints_to_replay_real(
            replay, task, "/tmp/replay", 0, 0, metadata, episode_keypoints, SCENE_BOUNDS,
            VOXEL_SIZES, ROTATION_RESOLUTION, False, 0, description="puck up can", clip_model=clip_model,
            episode_folder=EPISODE_PATH)

        self.assertEqual(len(replay.add_list), len(episode_keypoints))
        self.assertEqual(len(replay.add_final_list), 1)


if __name__ == "__main__":
    unittest.main()
