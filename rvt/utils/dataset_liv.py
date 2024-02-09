# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

# initial source: https://colab.research.google.com/drive/1HAqemP4cE81SQ6QO1-N85j5bF4C0qLs0?usp=sharing
# adapted to support loading from disk for faster initialization time

# Adapted from: https://github.com/stepjam/ARM/blob/main/arm/c2farm/launch_utils.py
import logging
import os
import pickle
from typing import List, Tuple

import clip
import numpy as np
import peract_colab.arm.utils as utils
import torch
import torchvision.transforms as T
from peract_colab.rlbench.utils import get_stored_demo
from rlbench.backend.observation import Observation
from rlbench.demo import Demo
from yarr.replay_buffer.replay_buffer import ReplayBuffer, ReplayElement
from yarr.replay_buffer.uniform_replay_buffer import UniformReplayBuffer
from yarr.utils.observation_type import ObservationElement

from rvt.libs.peract.helpers.demo_loading_utils import keypoint_discovery
from rvt.libs.peract.helpers.utils import extract_obs
from rvt.utils.dataset import _clip_encode_text, _get_action
from rvt.utils.peract_utils import CAMERAS, IMAGE_SIZE, LOW_DIM_SIZE


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
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                np.float32,
            )
        )
        observation_elements.append(
            ObservationElement(
                "%s_depth" % cname,
                (
                    1,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                np.float32,
            )
        )
        observation_elements.append(
            ObservationElement(
                "%s_point_cloud" % cname,
                (
                    3,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
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


# add individual data points to a replay
def _add_keypoints_to_replay(
    replay: ReplayBuffer,
    task: str,
    task_replay_storage_folder: str,
    episode_idx: int,
    sample_frame: int,
    inital_obs: Observation,
    demo: Demo,
    episode_keypoints: List[int],
    cameras: List[str],
    rlbench_scene_bounds: List[float],
    voxel_sizes: List[int],
    rotation_resolution: int,
    crop_augmentation: bool,
    next_keypoint_idx: int,
    description: str = "",
    clip_model=None,
    device="cpu",
    liv_model: torch.nn.Module=None,
    liv_embeddings=None,
):
    prev_action = None
    obs = inital_obs
    obs_i = sample_frame

    for k in range(
        next_keypoint_idx, len(episode_keypoints)
    ):  # confused here, it seems that there are many similar samples in the replay
        keypoint = episode_keypoints[k]
        obs_tp1 = demo[keypoint]
        obs_tm1 = demo[max(0, keypoint - 1)]
        (
            trans_indicies,
            rot_grip_indicies,
            ignore_collisions,
            action,
            attention_coordinates,
        ) = _get_action(
            obs_tp1,
            obs_tm1,
            rlbench_scene_bounds,
            voxel_sizes,
            rotation_resolution,
            crop_augmentation,
        )

        terminal = k == len(episode_keypoints) - 1
        reward = float(terminal) * 1.0 if terminal else 0

        obs_dict = extract_obs(
            obs,
            CAMERAS,
            t=k - next_keypoint_idx,
            prev_action=prev_action,
            episode_length=25,
        )
        tokens = clip.tokenize([description]).numpy()
        token_tensor = torch.from_numpy(tokens).to(device)
        with torch.no_grad():
            lang_feats, lang_embs = _clip_encode_text(clip_model, token_tensor)
        obs_dict["lang_goal_embs"] = lang_embs[0].float().detach().cpu().numpy()

        prev_action = np.copy(action)

        # Compute rewads based on state similarity.
        with torch.no_grad():
            s_obs_to_key_front = liv_model.sim(liv_embeddings[0][obs_i], liv_embeddings[0][keypoint]).detach().cpu().numpy()
            s_obs_to_goal_front = liv_model.sim(liv_embeddings[0][obs_i], liv_embeddings[0][-1]).detach().cpu().numpy()
            s_key_to_goal_front = liv_model.sim(liv_embeddings[0][keypoint], liv_embeddings[0][-1]).detach().cpu().numpy()

            s_obs_to_key_wrist = liv_model.sim(liv_embeddings[1][obs_i], liv_embeddings[1][keypoint]).detach().cpu().numpy()
            s_obs_to_goal_wrist = liv_model.sim(liv_embeddings[1][obs_i], liv_embeddings[1][-1]).detach().cpu().numpy()
            s_key_to_goal_wrist = liv_model.sim(liv_embeddings[1][keypoint], liv_embeddings[1][-1]).detach().cpu().numpy()

        # Reward to keypoint, reward to final goal, repeated for the front and wrist cameras.
        rewards = np.array([
            1. - s_obs_to_key_front,
            s_key_to_goal_front - s_obs_to_goal_front,
            1. - s_obs_to_key_wrist,
            s_key_to_goal_wrist - s_obs_to_goal_wrist,
        ], dtype=np.float32)

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
            "gripper_pose": obs_tp1.gripper_pose,
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
        obs = obs_tp1
        obs_i = keypoint
        sample_frame = keypoint

    # final step
    obs_dict_tp1 = extract_obs(
        obs_tp1,
        CAMERAS,
        t=k + 1 - next_keypoint_idx,
        prev_action=prev_action,
        episode_length=25,
    )
    obs_dict_tp1["lang_goal_embs"] = lang_embs[0].float().detach().cpu().numpy()

    obs_dict_tp1.pop("wrist_world_to_cam", None)
    # Not sure how this is used.
    final_obs["rewards"] = np.zeros((4,), dtype=np.float32)
    obs_dict_tp1.update(final_obs)
    replay.add_final(task, task_replay_storage_folder, **obs_dict_tp1)


@torch.no_grad()
def _get_liv_embeddings(
    demo, liv_model: torch.nn.Module, device, batch_size: int=64
    ):

    n = len(demo)

    front_imgs = []
    wrist_imgs = []

    transform = T.Compose([T.ToTensor()])

    for i in range(n):
        front_imgs.append(transform(demo[i].front_rgb))
        wrist_imgs.append(transform(demo[i].wrist_rgb))
    
    front_imgs = torch.stack(front_imgs)
    wrist_imgs = torch.stack(wrist_imgs)

    n_batches = int(np.ceil(n / batch_size))
    
    front_emb = []
    wrist_emb = []

    for i in range(n_batches):
        tmp_front = front_imgs[i * batch_size: (i + 1) * batch_size]
        tmp_wrist = wrist_imgs[i * batch_size: (i + 1) * batch_size]

        front_emb.append(
            liv_model(input=tmp_front.to(device), modality="vision").to("cpu")
        )
        wrist_emb.append(
            liv_model(input=tmp_wrist.to(device), modality="vision").to("cpu")
        )

    front_emb = torch.cat(front_emb)
    wrist_emb = torch.cat(wrist_emb)

    # Similarity calculation is done on CPU.
    front_emb = front_emb.float()
    wrist_emb = wrist_emb.float()

    return front_emb, wrist_emb


def fill_replay(
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
            demo = get_stored_demo(data_path=data_path, index=d_idx)

            # get language goal from disk
            varation_descs_pkl_file = os.path.join(
                data_path, episode_folder % d_idx, variation_desriptions_pkl
            )
            with open(varation_descs_pkl_file, "rb") as f:
                descs = pickle.load(f)

            # Compute LIV embeddings for all frames.
            liv_embeddings = _get_liv_embeddings(demo, liv_model, device)

            # extract keypoints
            episode_keypoints = keypoint_discovery(demo)
            next_keypoint_idx = 0
            for i in range(len(demo) - 1):
                if not demo_augmentation and i > 0:
                    break
                if i % demo_augmentation_every_n != 0:  # choose only every n-th frame
                    continue

                obs = demo[i]
                desc = descs[0]
                # if our starting point is past one of the keypoints, then remove it
                while (
                    next_keypoint_idx < len(episode_keypoints)
                    and i >= episode_keypoints[next_keypoint_idx]
                ):
                    next_keypoint_idx += 1
                if next_keypoint_idx == len(episode_keypoints):
                    break
                _add_keypoints_to_replay(
                    replay,
                    task,
                    task_replay_storage_folder,
                    d_idx,
                    i,
                    obs,
                    demo,
                    episode_keypoints,
                    cameras,
                    rlbench_scene_bounds,
                    voxel_sizes,
                    rotation_resolution,
                    crop_augmentation,
                    next_keypoint_idx=next_keypoint_idx,
                    description=desc,
                    clip_model=clip_model,
                    device=device,
                    liv_model=liv_model,
                    liv_embeddings=liv_embeddings,
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
