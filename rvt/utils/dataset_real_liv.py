# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

# initial source: https://colab.research.google.com/drive/1HAqemP4cE81SQ6QO1-N85j5bF4C0qLs0?usp=sharing
# adapted to support loading from disk for faster initialization time

# Adapted from: https://github.com/stepjam/ARM/blob/main/arm/c2farm/launch_utils.py
import os
import torch
import pickle
import logging
import numpy as np
from typing import Any, List

import clip
import peract_colab.arm.utils as utils
from PIL import Image
import torchvision.transforms as T

from yarr.utils.observation_type import ObservationElement
from yarr.replay_buffer.replay_buffer import ReplayElement, ReplayBuffer
from yarr.replay_buffer.uniform_replay_buffer import UniformReplayBuffer

from rvt.utils.peract_utils import LOW_DIM_SIZE, IMAGE_SIZE, CAMERAS

HEIGHT = 224
WIDTH = 298


class MockIntrinsics:
    def __init__(self, fx, fy, ppx, ppy):
        self.fx = fx
        self.fy = fy
        self.ppx = ppx
        self.ppy = ppy


def matrix_to_intrinsics(matrix):
    return MockIntrinsics(matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2])


def depth_to_pcd(
    depth, intrinsics: Any, depth_scale: float = 1000.0
):
    image_shape = depth.shape[:2]

    y_coords, x_coords = np.meshgrid(
        np.arange(image_shape[0]), np.arange(image_shape[1]), indexing="ij")
    pixels = np.stack((x_coords, y_coords), axis=-1).reshape(-1, 2)
    points = depth_pixels_to_pcd(depth, pixels, intrinsics, depth_scale=depth_scale)

    return np.array(points)


def depth_pixels_to_pcd(
    depth_points, pixels, intrinsics: Any,
    depth_scale: float = 1000.0
):
    if len(depth_points.shape) > 1:
        depth_points = depth_points.reshape(-1)

    z = depth_points / depth_scale
    x = (pixels[:, 0] - intrinsics.ppx) * z / intrinsics.fx
    y = (pixels[:, 1] - intrinsics.ppy) * z / intrinsics.fy

    return np.stack([x, y, z], -1)


def transform_pcd(pcd, T):

    assert T.shape == (4, 4), "Expect a 3D homogeneous transform."

    if len(pcd) == 0:
        return pcd

    pcd = np.concatenate([pcd, np.ones_like(pcd)[..., 0: 1]], axis=-1)
    pcd = pcd @ T.T
    pcd = pcd[..., :-1]
    return pcd


def extract_obs_real(obs_idx: int,
                     metadata: dict,
                     t: int = 0,
                     channels_last: bool = False,
                     episode_length: int = 10,
                     episode_folder: str = ""):

    # Image and depth size should be 128**2. Let's keep the original size for now.

    rgb1_path = os.path.join(episode_folder, "camera1_rgb", f"{obs_idx}.png")
    rgb2_path = os.path.join(episode_folder, "camera2_rgb", f"{obs_idx}.png")
    rgb3_path = os.path.join(episode_folder, "camera3_rgb", f"{obs_idx}.png")

    depth1_path = os.path.join(episode_folder, "camera1_depth", f"{obs_idx}.npy")
    depth2_path = os.path.join(episode_folder, "camera2_depth", f"{obs_idx}.npy")
    depth3_path = os.path.join(episode_folder, "camera3_depth", f"{obs_idx}.npy")

    with Image.open(rgb1_path) as img:
        rgb1 = np.array(img, dtype=np.uint8)
    
    with Image.open(rgb2_path) as img:
        rgb2 = np.array(img, dtype=np.uint8)

    with Image.open(rgb3_path) as img:
        rgb3 = np.array(img, dtype=np.uint8)

    depth1 = np.load(depth1_path)
    depth2 = np.load(depth2_path)
    depth3 = np.load(depth3_path)

    rgb1 = rgb1.transpose((2, 0, 1))
    rgb2 = rgb2.transpose((2, 0, 1))
    rgb3 = rgb3.transpose((2, 0, 1))

    pcd1 = depth_to_pcd(depth1, matrix_to_intrinsics(metadata["int1"]))
    pcd2 = depth_to_pcd(depth2, matrix_to_intrinsics(metadata["int2"]))
    pcd3 = depth_to_pcd(depth3, matrix_to_intrinsics(metadata["int3"]))

    pcd1 = transform_pcd(pcd1, metadata["ext1"])
    pcd2 = transform_pcd(pcd2, metadata["ext2"])
    pcd3 = transform_pcd(pcd3, metadata["ext3"])

    pcd1 = pcd1.reshape((HEIGHT, WIDTH, 3))
    pcd2 = pcd2.reshape((HEIGHT, WIDTH, 3))
    pcd3 = pcd3.reshape((HEIGHT, WIDTH, 3))

    pcd1 = pcd1.transpose((2, 0, 1))
    pcd2 = pcd2.transpose((2, 0, 1))
    pcd3 = pcd3.transpose((2, 0, 1))

    depth1 = depth1[None]
    depth2 = depth2[None]
    depth3 = depth3[None]

    rgb_ref_shape = (3, HEIGHT, WIDTH)
    depth_ref_shape = (1, HEIGHT, WIDTH)
    pcd_ref_shape = (3, HEIGHT, WIDTH)

    assert rgb1.shape == rgb_ref_shape
    assert rgb2.shape == rgb_ref_shape
    assert rgb3.shape == rgb_ref_shape

    assert depth1.shape == depth_ref_shape
    assert depth2.shape == depth_ref_shape
    assert depth3.shape == depth_ref_shape

    assert pcd1.shape == pcd_ref_shape
    assert pcd2.shape == pcd_ref_shape
    assert pcd3.shape == pcd_ref_shape

    obs_dict = {}
    obs_dict["camera1_rgb"] = rgb1
    obs_dict["camera2_rgb"] = rgb2
    obs_dict["camera3_rgb"] = rgb3

    obs_dict["camera1_depth"] = depth1
    obs_dict["camera2_depth"] = depth2
    obs_dict["camera3_depth"] = depth3

    obs_dict["camera1_point_cloud"] = pcd1
    obs_dict["camera2_point_cloud"] = pcd2
    obs_dict["camera3_point_cloud"] = pcd3

    time = (1. - (t / float(episode_length - 1))) * 2. - 1.
    low_dim_state = np.array(
        [metadata["gripper_open"][obs_idx], *metadata["gripper_q"][obs_idx], time])
    assert low_dim_state.shape == (4,)

    obs_dict["low_dim_state"] = low_dim_state

    obs_dict["camera1_camera_intrinsics"] = metadata["int1"]
    obs_dict["camera2_camera_intrinsics"] = metadata["int2"]
    obs_dict["camera3_camera_intrinsics"] = metadata["int3"]

    obs_dict["camera1_camera_extrinsics"] = metadata["ext1"]
    obs_dict["camera2_camera_extrinsics"] = metadata["ext2"]
    obs_dict["camera3_camera_extrinsics"] = metadata["ext3"]

    obs_dict["ignore_collisions"] = metadata["ignore_collisions"][obs_idx: obs_idx + 1]

    return obs_dict


def _is_stopped(metadata: dict, i: int, stopped_buffer: int, delta: float=0.1):
    next_is_not_final = i == (len(metadata["gripper_open"]) - 2)
    gripper_state_no_change = (
            i < (len(metadata["gripper_open"]) - 2) and
            (metadata["gripper_open"][i] == metadata["gripper_open"][i + 1] and
             metadata["gripper_open"][i] == metadata["gripper_open"][i - 1] and
             metadata["gripper_open"][i - 2] == metadata["gripper_open"][i - 1]))
    small_delta = np.allclose(metadata["joint_velocities"][i], 0, atol=delta)
    stopped = (stopped_buffer <= 0 and small_delta and
               (not next_is_not_final) and gripper_state_no_change)
    return stopped


def keypoint_discovery_real(metadata: dict,
                            stopping_delta: float=0.01,
                            max_stopped_frames: int=30,
                            method='heuristic') -> List[int]:
    episode_keypoints = []
    if method == 'heuristic':
        prev_gripper_open = metadata["gripper_open"][0]
        stopped_buffer = 0
        for i in range(len(metadata["gripper_open"])):
            stopped = _is_stopped(metadata, i, stopped_buffer, stopping_delta)
            stopped_buffer = max_stopped_frames if stopped else stopped_buffer - 1
            # If change in gripper, or end of episode.
            last = i == (len(metadata["gripper_open"]) - 1)
            if i != 0 and (metadata["gripper_open"][i] != prev_gripper_open or
                           last or stopped):
                episode_keypoints.append(i)
            prev_gripper_open = metadata["gripper_open"][i]
        if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == \
                episode_keypoints[-2]:
            episode_keypoints.pop(-2)
        logging.debug('Found %d keypoints.' % len(episode_keypoints),
                      episode_keypoints)
        return episode_keypoints

    elif method == 'random':
        # Randomly select keypoints.
        episode_keypoints = np.random.choice(
            range(len(metadata["gripper_open"])),
            size=20,
            replace=False)
        episode_keypoints.sort()
        return episode_keypoints

    elif method == 'fixed_interval':
        # Fixed interval.
        episode_keypoints = []
        segment_length = len(metadata["gripper_open"]) // 20
        for i in range(0, len(metadata["gripper_open"]), segment_length):
            episode_keypoints.append(i)
        return episode_keypoints

    else:
        raise NotImplementedError


def create_replay(
    batch_size: int,
    timesteps: int,
    disk_saving: bool,
    cameras: list,
    voxel_sizes,
    replay_size=3e5,
):

    trans_indicies_size = 3 * len(voxel_sizes)
    rot_and_grip_indicies_size = 3 + 1
    gripper_pose_size = 7
    ignore_collisions_size = 1
    max_token_seq_len = 77
    lang_feat_dim = 1024
    lang_emb_dim = 512

    # low_dim_state
    observation_elements = []
    observation_elements.append(
        ObservationElement("low_dim_state", (LOW_DIM_SIZE,), np.float32)
    )

    # rgb, depth, point cloud, intrinsics, extrinsics
    for cname in cameras:
        observation_elements.append(
            ObservationElement(
                "%s_rgb" % cname,
                (
                    3,
                    HEIGHT,
                    WIDTH,
                ),
                np.float32,
            )
        )
        observation_elements.append(
            ObservationElement(
                "%s_depth" % cname,
                (
                    1,
                    HEIGHT,
                    WIDTH,
                ),
                np.float32,
            )
        )
        observation_elements.append(
            ObservationElement(
                "%s_point_cloud" % cname,
                (
                    3,
                    HEIGHT,
                    WIDTH,
                ),
                np.float32,
            )
        )  # see pyrep/objects/vision_sensor.py on how pointclouds are extracted from depth frames
        observation_elements.append(
            ObservationElement(
                "%s_camera_extrinsics" % cname,
                (
                    4,
                    4,
                ),
                np.float32,
            )
        )
        observation_elements.append(
            ObservationElement(
                "%s_camera_intrinsics" % cname,
                (
                    3,
                    3,
                ),
                np.float32,
            )
        )

    # discretized translation, discretized rotation, discrete ignore collision, 6-DoF gripper pose, and pre-trained language embeddings
    observation_elements.extend(
        [
            ReplayElement("trans_action_indicies", (trans_indicies_size,), np.int32),
            ReplayElement(
                "rot_grip_action_indicies", (rot_and_grip_indicies_size,), np.int32
            ),
            ReplayElement("ignore_collisions", (ignore_collisions_size,), np.int32),
            ReplayElement("gripper_pose", (gripper_pose_size,), np.float32),
            ReplayElement(
                "lang_goal_embs",
                (
                    max_token_seq_len,
                    lang_emb_dim,
                ),  # extracted from CLIP's language encoder
                np.float32,
            ),
            ReplayElement(
                "lang_goal", (1,), object
            ),  # language goal string for debugging and visualization
            ReplayElement("rewards", (4,), np.float32),
        ]
    )

    extra_replay_elements = [
        ReplayElement("demo", (), bool),
        ReplayElement("keypoint_idx", (), int),
        ReplayElement("episode_idx", (), int),
        ReplayElement("keypoint_frame", (), int),
        ReplayElement("next_keypoint_frame", (), int),
        ReplayElement("sample_frame", (), int),
    ]

    replay_buffer = (
        UniformReplayBuffer(  # all tuples in the buffer have equal sample weighting
            disk_saving=disk_saving,
            batch_size=batch_size,
            timesteps=timesteps,
            replay_capacity=int(replay_size),
            action_shape=(8,),  # 3 translation + 4 rotation quaternion + 1 gripper open
            action_dtype=np.float32,
            reward_shape=(),
            reward_dtype=np.float32,
            update_horizon=1,
            observation_elements=observation_elements,
            extra_replay_elements=extra_replay_elements,
        )
    )
    return replay_buffer


# discretize translation, rotation, gripper open, and ignore collision actions
def _get_action(
    tp1: int,
    tm1: int,
    metadata: dict,
    rlbench_scene_bounds: List[float],  # metric 3D bounds of the scene
    voxel_sizes: List[int],
    rotation_resolution: int,
    crop_augmentation: bool,
):
    quat = utils.normalize_quaternion(metadata["gripper_pose"][tp1][3:])
    if quat[-1] < 0:
        quat = -quat
    disc_rot = utils.quaternion_to_discrete_euler(quat, rotation_resolution)
    attention_coordinate = metadata["gripper_pose"][tp1][:3]
    trans_indicies, attention_coordinates = [], []
    bounds = np.array(rlbench_scene_bounds)
    ignore_collisions = int(metadata["ignore_collisions"][tm1])
    for depth, vox_size in enumerate(
        voxel_sizes
    ):  # only single voxelization-level is used in PerAct
        index = utils.point_to_voxel_index(metadata["gripper_pose"][tp1][:3], vox_size, bounds)
        trans_indicies.extend(index.tolist())
        res = (bounds[3:] - bounds[:3]) / vox_size
        attention_coordinate = bounds[:3] + res * index
        attention_coordinates.append(attention_coordinate)

    rot_and_grip_indicies = disc_rot.tolist()
    grip = float(metadata["gripper_open"][tp1])
    rot_and_grip_indicies.extend([int(metadata["gripper_open"][tp1])])
    return (
        trans_indicies,
        rot_and_grip_indicies,
        ignore_collisions,
        np.concatenate([metadata["gripper_pose"][tp1], np.array([grip])]),
        attention_coordinates,
    )


# extract CLIP language features for goal string
def _clip_encode_text(clip_model, text):
    x = clip_model.token_embedding(text).type(
        clip_model.dtype
    )  # [batch_size, n_ctx, d_model]

    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(clip_model.dtype)

    emb = x.clone()
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ clip_model.text_projection

    return x, emb


# add individual data points to a replay
def _add_keypoints_to_replay_real(
    replay: ReplayBuffer,
    task: str,
    task_replay_storage_folder: str,
    episode_idx: int,
    sample_frame: int,
    metadata: dict,
    episode_keypoints: List[int],
    rlbench_scene_bounds: List[float],
    voxel_sizes: List[int],
    rotation_resolution: int,
    crop_augmentation: bool,
    next_keypoint_idx: int,
    description: str = "",
    clip_model=None,
    device="cpu",
    liv_model=None,
    liv_embeddings_image=None,
    liv_embedding_text=None,
    episode_folder="",
):
    prev_action = None
    obs_idx = sample_frame
    for k in range(
        next_keypoint_idx, len(episode_keypoints)
    ):  # confused here, it seems that there are many similar samples in the replay
        keypoint = episode_keypoints[k]
        tp1 = keypoint
        tm1 = max(0, keypoint - 1)
        (
            trans_indicies,
            rot_grip_indicies,
            ignore_collisions,
            action,
            attention_coordinates,
        ) = _get_action(
            tp1,
            tm1,
            metadata,
            rlbench_scene_bounds,
            voxel_sizes,
            rotation_resolution,
            crop_augmentation,
        )

        terminal = k == len(episode_keypoints) - 1
        reward = float(terminal) * 1.0 if terminal else 0

        obs_dict = extract_obs_real(
            obs_idx,
            metadata,
            t=k - next_keypoint_idx,
            episode_length=25,
            episode_folder=episode_folder
        )
        tokens = clip.tokenize([description]).numpy()
        token_tensor = torch.from_numpy(tokens).to(device)
        with torch.no_grad():
            lang_feats, lang_embs = _clip_encode_text(clip_model, token_tensor)
        obs_dict["lang_goal_embs"] = lang_embs[0].float().detach().cpu().numpy()

        prev_action = np.copy(action)

        with torch.no_grad():
            # Look one second (15 fps) after the keypoint to make sure the world has settled.
            # E.g. dropping an item into a container.
            tmp_keypoint = min(len(liv_embeddings_image) - 1, keypoint + 15)
            s_keypoint_to_text = liv_model.sim(liv_embeddings_image[tmp_keypoint], liv_embedding_text).detach().cpu().numpy()
            s_current_to_text = liv_model.sim(liv_embeddings_image[obs_idx], liv_embedding_text).detach().cpu().numpy()
            s_final_to_text = liv_model.sim(liv_embeddings_image[-1], liv_embedding_text).detach().cpu().numpy()

            s_obs_to_key = liv_model.sim(liv_embeddings_image[obs_idx], liv_embeddings_image[tmp_keypoint]).detach().cpu().numpy()
            s_obs_to_goal = liv_model.sim(liv_embeddings_image[obs_idx], liv_embeddings_image[-1]).detach().cpu().numpy()
            s_key_to_goal = liv_model.sim(liv_embeddings_image[tmp_keypoint], liv_embeddings_image[-1]).detach().cpu().numpy()

        rewards = np.array([
            s_final_to_text - 1.,
            s_keypoint_to_text - s_current_to_text,
            1. - s_obs_to_key,
            s_key_to_goal - s_obs_to_goal,
        ])

        if k == 0:
            keypoint_frame = -1
        else:
            keypoint_frame = episode_keypoints[k - 1]
        others = {
            "demo": True,
            "keypoint_idx": k,
            "episode_idx": episode_idx,
            "keypoint_frame": keypoint_frame,
            "next_keypoint_frame": keypoint,
            "sample_frame": sample_frame,
            "rewards": rewards,
        }
        final_obs = {
            "trans_action_indicies": trans_indicies,
            "rot_grip_action_indicies": rot_grip_indicies,
            "gripper_pose": metadata["gripper_pose"][tp1],
            "lang_goal": np.array([description], dtype=object),
        }

        others.update(final_obs)
        others.update(obs_dict)

        timeout = False
        replay.add(
            task,
            task_replay_storage_folder,
            action,
            reward,
            terminal,
            timeout,
            **others
        )
        obs_idx = tp1
        sample_frame = keypoint

    # final step
    obs_dict_tp1 = extract_obs_real(
        tp1,
        metadata,
        t=k + 1 - next_keypoint_idx,
        episode_length=25,
        episode_folder=episode_folder
    )
    obs_dict_tp1["lang_goal_embs"] = lang_embs[0].float().detach().cpu().numpy()
    obs_dict_tp1.pop("wrist_world_to_cam", None)
    # Not sure how this is used.
    final_obs["rewards"] = np.zeros((4,), dtype=np.float32)
    obs_dict_tp1.update(final_obs)
    replay.add_final(task, task_replay_storage_folder, **obs_dict_tp1)


@torch.no_grad()
def _get_liv_embeddings(
    episode_path, description, liv_model: torch.nn.Module, device, batch_size: int=64
    ):
    
    tmp_path = os.path.join(episode_path, f"camera2_rgb")
    image_paths = [f"{i}.png" for i in range(len(os.listdir(tmp_path)))]
    image_paths = [os.path.join(tmp_path, p) for p in image_paths]

    transform = T.Compose([T.ToTensor()])
    image_tensors = []
    for image_path in image_paths:
        with Image.open(image_path) as img:
            image_tensors.append(transform(img))
    image_tensors = torch.stack(image_tensors)

    embeddings = liv_model(input=image_tensors.to(device), modality="vision")
    token = clip.tokenize([description])
    goal_embedding_text = liv_model(input=token.to(device), modality="text")
    goal_embedding_text = goal_embedding_text[0] 

    embeddings = embeddings.float()
    goal_embedding_text = goal_embedding_text.float()

    return embeddings, goal_embedding_text


def fill_replay_real(
    replay: ReplayBuffer,
    task: str,
    task_replay_storage_folder: str,
    start_idx: int,
    num_demos: int,
    demo_augmentation: bool,
    demo_augmentation_every_n: int,
    cameras: List[str],
    rlbench_scene_bounds: List[float],  # AKA: DEPTH0_BOUNDS
    voxel_sizes: List[int],
    rotation_resolution: int,
    crop_augmentation: bool,
    data_path: str,
    episode_folder: str,
    variation_desriptions_pkl: str,
    clip_model=None,
    liv_model=None,
    device="cpu",
):

    disk_exist = False
    if replay._disk_saving:
        if os.path.exists(task_replay_storage_folder):
            print(
                "[Info] Replay dataset already exists in the disk: {}".format(
                    task_replay_storage_folder
                ),
                flush=True,
            )
            disk_exist = True
        else:
            logging.info("\t saving to disk: %s", task_replay_storage_folder)
            os.makedirs(task_replay_storage_folder, exist_ok=True)

    if disk_exist:
        replay.recover_from_disk(task, task_replay_storage_folder)
    else:
        print("Filling replay ...")
        for d_idx in range(start_idx, start_idx + num_demos):
            print("Filling demo %d" % d_idx)

            tmp_data_path = os.path.join(data_path, f"episode{d_idx}")

            if not os.path.exists(tmp_data_path):
                print(f"WARNING: task {task} has only {d_idx - start_idx} demos")
                break

            metadata_path = os.path.join(tmp_data_path, "metadata.pkl")
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)

            # Compute LIV embeddings for all frames.
            liv_embeddings_image, liv_embedding_text = _get_liv_embeddings(tmp_data_path, metadata["description"], liv_model, device)

            # extract keypoints
            episode_keypoints = keypoint_discovery_real(metadata)
            next_keypoint_idx = 0
            for i in range(len(metadata["gripper_open"]) - 1):
                if not demo_augmentation and i > 0:
                    break
                if i % demo_augmentation_every_n != 0:  # choose only every n-th frame
                    continue

                # if our starting point is past one of the keypoints, then remove it
                while (
                    next_keypoint_idx < len(episode_keypoints)
                    and i >= episode_keypoints[next_keypoint_idx]
                ):
                    next_keypoint_idx += 1
                if next_keypoint_idx == len(episode_keypoints):
                    break
                _add_keypoints_to_replay_real(
                    replay,
                    task,
                    task_replay_storage_folder,
                    d_idx,
                    i,
                    metadata,
                    episode_keypoints,
                    rlbench_scene_bounds,
                    voxel_sizes,
                    rotation_resolution,
                    crop_augmentation,
                    next_keypoint_idx=next_keypoint_idx,
                    description=metadata["description"],
                    clip_model=clip_model,
                    device=device,
                    liv_model=liv_model,
                    liv_embeddings_image=liv_embeddings_image,
                    liv_embedding_text=liv_embedding_text,
                    episode_folder=tmp_data_path
                )

        # save TERMINAL info in replay_info.npy
        task_idx = replay._task_index[task]
        with open(
            os.path.join(task_replay_storage_folder, "replay_info.npy"), "wb"
        ) as fp:
            np.save(
                fp,
                replay._store["terminal"][
                    replay._task_replay_start_index[
                        task_idx
                    ] : replay._task_replay_start_index[task_idx]
                    + replay._task_add_count[task_idx].value
                ],
            )

        print("Replay filled with demos.")
