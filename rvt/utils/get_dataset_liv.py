# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].
import os
import shutil

import clip
import torch
from rvt.utils.dataset_liv import create_replay, fill_replay
from rvt.utils.peract_utils import (
    CAMERAS,
    DEMO_AUGMENTATION_EVERY_N,
    EPISODE_FOLDER,
    ROTATION_RESOLUTION,
    SCENE_BOUNDS,
    VARIATION_DESCRIPTIONS_PKL,
    VOXEL_SIZES,
)
from torch import nn
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import PyTorchReplayBuffer


def convert_weights_to_full(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_full(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.float()
            if l.bias is not None:
                l.bias.data = l.bias.data.float()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.float()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.float()

    model.apply(_convert_weights_to_full)


def cleanup_config(cfg):

    import copy

    config = copy.deepcopy(cfg)
    config["device"] = "cpu"

    return config.agent


def load_liv(modelid='resnet50', new_checkpoint: bool=False):

    import os
    from os.path import expanduser

    import gdown
    import hydra
    import omegaconf
    import torch
    from huggingface_hub import hf_hub_download

    assert modelid == 'resnet50'
    home = os.path.join(expanduser("~"), ".liv")

    if not os.path.exists(os.path.join(home, modelid)):
        os.makedirs(os.path.join(home, modelid))
    folderpath = os.path.join(home, modelid)
    modelpath = os.path.join(home, modelid, "model.pt")
    configpath = os.path.join(home, modelid, "config.yaml")

    if not os.path.exists(modelpath):
        try:
            # Default reliable download from HuggingFace Hub
            hf_hub_download(repo_id="jasonyma/LIV", filename="model.pt", local_dir=folderpath)
            hf_hub_download(repo_id="jasonyma/LIV", filename="config.yaml", local_dir=folderpath)
        except:
            # Download from GDown
            modelurl = 'https://drive.google.com/uc?id=1l1ufzVLxpE5BK7JY6ZnVBljVzmK5c4P3'
            configurl = 'https://drive.google.com/uc?id=1GWA5oSJDuHGB2WEdyZZmkro83FNmtaWl'
            gdown.download(modelurl, modelpath, quiet=False)
            gdown.download(configurl, configpath, quiet=False)

    modelcfg = omegaconf.OmegaConf.load(configpath)
    cleancfg = cleanup_config(modelcfg)
    rep = hydra.utils.instantiate(cleancfg)
    state_dict = torch.load(modelpath, map_location=torch.device("cpu"))['liv']
    
    # Remove "module." because we are not using data parallelism.
    to_remove = "module."
    state_dict = dict(state_dict)
    keys = list(state_dict.keys())
    for key in keys:
        if key[:len(to_remove)] == to_remove:
            new_key = key[len(to_remove):]
            state_dict[new_key] = state_dict[key]
            del state_dict[key]

    if new_checkpoint:
        # TODO: Not sure why I need this for new checkpoints.
        state_dict = dict(state_dict)
        keys = list(state_dict.keys())
        for key in keys:
            state_dict["module." + key] = state_dict[key]
            del state_dict[key]

    rep.load_state_dict(state_dict)
    return rep


def get_dataset(
    tasks,
    BATCH_SIZE_TRAIN,
    BATCH_SIZE_TEST,
    TRAIN_REPLAY_STORAGE_DIR,
    TEST_REPLAY_STORAGE_DIR,
    DATA_FOLDER,
    NUM_TRAIN,
    NUM_VAL,
    refresh_replay,
    device,
    num_workers,
    only_train,
    sample_distribution_mode="transition_uniform",
):

    train_replay_buffer = create_replay(
        batch_size=BATCH_SIZE_TRAIN,
        timesteps=1,
        disk_saving=True,
        cameras=CAMERAS,
        voxel_sizes=VOXEL_SIZES,
    )
    if not only_train:
        test_replay_buffer = create_replay(
            batch_size=BATCH_SIZE_TEST,
            timesteps=1,
            disk_saving=True,
            cameras=CAMERAS,
            voxel_sizes=VOXEL_SIZES,
        )

    # load pre-trained language model
    try:
        clip_model, _ = clip.load("RN50", device="cpu")  # CLIP-ResNet50
        clip_model = clip_model.to(device)
        clip_model.eval()
    except RuntimeError:
        print("WARNING: Setting Clip to None. Will not work if replay not on disk.")
        clip_model = None

    try:
        # Load pre-trained LIV model.
        liv_model = load_liv()
        liv_model = liv_model.to(device)
        liv_model.device = device
        liv_model.eval()
    except RuntimeError as e:
        print(e)
        print("WARNING: Setting LIV to None. Will not work if replay not on disk.")
        liv_model = None

    for task in tasks:  # for each task
        # print("---- Preparing the data for {} task ----".format(task), flush=True)
        EPISODES_FOLDER_TRAIN = f"train/{task}/all_variations/episodes"
        EPISODES_FOLDER_VAL = f"val/{task}/all_variations/episodes"
        data_path_train = os.path.join(DATA_FOLDER, EPISODES_FOLDER_TRAIN)
        data_path_val = os.path.join(DATA_FOLDER, EPISODES_FOLDER_VAL)
        train_replay_storage_folder = f"{TRAIN_REPLAY_STORAGE_DIR}/{task}"
        test_replay_storage_folder = f"{TEST_REPLAY_STORAGE_DIR}/{task}"

        # if refresh_replay, then remove the existing replay data folder
        if refresh_replay:
            print("[Info] Remove exisitng replay dataset as requested.", flush=True)
            if os.path.exists(train_replay_storage_folder) and os.path.isdir(
                train_replay_storage_folder
            ):
                shutil.rmtree(train_replay_storage_folder)
                print(f"remove {train_replay_storage_folder}")
            if os.path.exists(test_replay_storage_folder) and os.path.isdir(
                test_replay_storage_folder
            ):
                shutil.rmtree(test_replay_storage_folder)
                print(f"remove {test_replay_storage_folder}")

        # print("----- Train Buffer -----")
        fill_replay(
            replay=train_replay_buffer,
            task=task,
            task_replay_storage_folder=train_replay_storage_folder,
            start_idx=0,
            num_demos=NUM_TRAIN,
            demo_augmentation=True,
            demo_augmentation_every_n=DEMO_AUGMENTATION_EVERY_N,
            cameras=CAMERAS,
            rlbench_scene_bounds=SCENE_BOUNDS,
            voxel_sizes=VOXEL_SIZES,
            rotation_resolution=ROTATION_RESOLUTION,
            crop_augmentation=False,
            data_path=data_path_train,
            episode_folder=EPISODE_FOLDER,
            variation_desriptions_pkl=VARIATION_DESCRIPTIONS_PKL,
            clip_model=clip_model,
            liv_model=liv_model,
            device=device,
        )

        if not only_train:
            # print("----- Test Buffer -----")
            fill_replay(
                replay=test_replay_buffer,
                task=task,
                task_replay_storage_folder=test_replay_storage_folder,
                start_idx=0,
                num_demos=NUM_VAL,
                demo_augmentation=True,
                demo_augmentation_every_n=DEMO_AUGMENTATION_EVERY_N,
                cameras=CAMERAS,
                rlbench_scene_bounds=SCENE_BOUNDS,
                voxel_sizes=VOXEL_SIZES,
                rotation_resolution=ROTATION_RESOLUTION,
                crop_augmentation=False,
                data_path=data_path_val,
                episode_folder=EPISODE_FOLDER,
                variation_desriptions_pkl=VARIATION_DESCRIPTIONS_PKL,
                clip_model=clip_model,
                liv_model=liv_model,
                device=device,
            )

    # delete the CLIP model since we have already extracted language features
    del clip_model
    del liv_model
    with torch.cuda.device(device):
        torch.cuda.empty_cache()

    # wrap buffer with PyTorch dataset and make iterator
    train_wrapped_replay = PyTorchReplayBuffer(
        train_replay_buffer,
        sample_mode="random",
        num_workers=num_workers,
        sample_distribution_mode=sample_distribution_mode,
    )
    train_dataset = train_wrapped_replay.dataset()

    if only_train:
        test_dataset = None
    else:
        test_wrapped_replay = PyTorchReplayBuffer(
            test_replay_buffer,
            sample_mode="enumerate",
            num_workers=num_workers,
        )
        test_dataset = test_wrapped_replay.dataset()
    return train_dataset, test_dataset
