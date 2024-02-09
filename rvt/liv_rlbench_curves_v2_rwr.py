import argparse
import os
import pickle

import clip
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import numpy as np
import torch
import torchvision.transforms as T
from liv import load_liv
from PIL import Image
from video_imitation import utils
import cv2

from rvt.libs.peract.helpers.demo_loading_utils import keypoint_discovery

CAMERAS = ["front", "left_shoulder", "overhead", "right_shoulder", "wrist"]


def find_next_keypoint(t, keypoints):
    for keypoint in keypoints:
        if keypoint > t:
            return keypoint
    return keypoints[-1]


def plot_curve(embeddings, keypoints, goal_embedding_text, task, model, imgs_tensor, save_dir, name, tau=0.1):

    fig, ax = plt.subplots(nrows=1, ncols=4,figsize=(24, 6))
    distances_cur_img = []
    distances_cur_text = [] 

    for t in range(embeddings.shape[0]):
        cur_embedding = embeddings[t]
        cur_distance_img = 1. - model.module.sim(embeddings[find_next_keypoint(t, keypoints)], cur_embedding).detach().cpu().numpy()
        cur_distance_text = 1. - model.module.sim(goal_embedding_text, cur_embedding).detach().cpu().numpy()

        cur_distance_img = np.exp(tau * cur_distance_img)
        cur_distance_text = np.exp(tau * cur_distance_text)

        distances_cur_img.append(cur_distance_img)
        distances_cur_text.append(cur_distance_text)

    distances_cur_img = np.array(distances_cur_img)
    distances_cur_text = np.array(distances_cur_text)

    ax[0].plot(np.arange(len(distances_cur_img)), distances_cur_img, color="tab:blue", label="image", linewidth=3)
    ax[0].plot(np.arange(len(distances_cur_text)), distances_cur_text, color="tab:red", label="text", linewidth=3)
    ax[1].plot(np.arange(len(distances_cur_img)), distances_cur_img, color="tab:blue", label="image", linewidth=3)
    ax[2].plot(np.arange(len(distances_cur_text)), distances_cur_text, color="tab:red", label="text", linewidth=3)
    ax[1].scatter(keypoints, -np.ones_like(keypoints), color="tab:red")

    ax[0].legend(loc="upper right")
    ax[0].set_xlabel("Frame", fontsize=15)
    ax[1].set_xlabel("Frame", fontsize=15)
    ax[2].set_xlabel("Frame", fontsize=15)
    ax[0].set_ylabel("Embedding Distance", fontsize=15)
    ax[0].set_title(f"Language Goal: {task}", fontsize=15)
    ax[1].set_title("Image Goal", fontsize=15)
    ax[2].set_title(f"Language Goal: {task}", fontsize=15)
    ax[3].imshow(imgs_tensor[-1].permute(1,2,0))
    asp = 1
    ax[0].set_aspect(asp * np.diff(ax[0].get_xlim())[0] / np.diff(ax[0].get_ylim())[0])
    ax[1].set_aspect(asp * np.diff(ax[1].get_xlim())[0] / np.diff(ax[1].get_ylim())[0])
    ax[2].set_aspect(asp * np.diff(ax[2].get_xlim())[0] / np.diff(ax[2].get_ylim())[0])
    fig.savefig(os.path.join(save_dir, f"{name}.png"), bbox_inches='tight')
    plt.close()

    ax0_xlim = ax[0].get_xlim()
    ax0_ylim = ax[0].get_ylim()
    ax1_xlim = ax[1].get_xlim()
    ax1_ylim = ax[1].get_ylim()
    ax2_xlim = ax[2].get_xlim()
    ax2_ylim = ax[2].get_ylim()

    # def animate(i):
    #     for ax_subplot in ax:
    #         ax_subplot.clear()
    #     ranges = np.arange(len(distances_cur_img))
    #     if i >= len(distances_cur_img):
    #         i = len(distances_cur_img) - 1
    #     line1 = ax[0].plot(ranges[0:i], distances_cur_img[0:i], color="tab:blue", label="image", linewidth=3)
    #     line2 = ax[0].plot(ranges[0:i], distances_cur_text[0:i], color="tab:red", label="text", linewidth=3)
    #     line3 = ax[1].plot(ranges[0:i], distances_cur_img[0:i], color="tab:blue", label="image", linewidth=3)
    #     line4 = ax[2].plot(ranges[0:i], distances_cur_text[0:i], color="tab:red", label="text", linewidth=3)
    #     line5 = ax[3].imshow(imgs_tensor[i].permute(1,2,0))
    #     line6 = ax[1].scatter(keypoints, -np.ones_like(keypoints), color="tab:red")

    #     ax[0].legend(loc="upper right")
    #     ax[0].set_xlabel("Frame", fontsize=15)
    #     ax[1].set_xlabel("Frame", fontsize=15)
    #     ax[2].set_xlabel("Frame", fontsize=15)
    #     ax[0].set_ylabel("Embedding Distance", fontsize=15)
    #     ax[0].set_title(f"Language Goal: {task}", fontsize=15)
    #     ax[1].set_title("Image Goal", fontsize=15)
    #     ax[2].set_title(f"Language Goal: {task}", fontsize=15)

    #     ax[0].set_xlim(ax0_xlim)
    #     ax[0].set_ylim(ax0_ylim)
    #     ax[1].set_xlim(ax1_xlim)
    #     ax[1].set_ylim(ax1_ylim)
    #     ax[2].set_xlim(ax2_xlim)
    #     ax[2].set_ylim(ax2_ylim)

    #     return line1, line2, line3, line4, line5, line6

    # # Generate animated reward curve
    # ani = FuncAnimation(fig, animate, interval=20, repeat=False, frames=len(distances_cur_img)+30)
    # ani.save(os.path.join(save_dir, f"{name}.gif"), dpi=100, writer=PillowWriter(fps=25))


def main(args):

    load_path = f"{args.task_path}/all_variations/episodes/episode{args.episode}"
    save_path = f"/home/obiza/reward_curves_keyframes_rwr/tau_{args.tau}/{args.task_path.split('/')[-1]}/episode{args.episode}"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # loading LIV
    liv = load_liv()
    liv.eval()
    transform = T.Compose([T.ToTensor()])

    with open(os.path.join(load_path, "low_dim_obs.pkl"), "rb") as f:
        low_dim_obs = pickle.load(f)
    
    with open(os.path.join(load_path, "variation_descriptions.pkl"), "rb") as f:
        variation_descriptions = pickle.load(f)
    
    text = variation_descriptions[0]
    keypoints = keypoint_discovery(low_dim_obs)

    for camera in CAMERAS:
        tmp_path = os.path.join(load_path, f"{camera}_rgb")
        image_paths = [f"{i}.png" for i in range(len(os.listdir(tmp_path)))]
        image_paths = [os.path.join(tmp_path, p) for p in image_paths]

        image_tensors = []
        for image_path in image_paths:
            with Image.open(image_path) as img:
                image_tensors.append(transform(img))
        image_tensors = torch.stack(image_tensors)

        with torch.no_grad():
            embeddings = liv(input=image_tensors.to(device), modality="vision")
            token = clip.tokenize([text])
            goal_embedding_text = liv(input=token, modality="text")
            goal_embedding_text = goal_embedding_text[0] 

        os.makedirs(save_path, exist_ok=True)
        plot_curve(
            embeddings, keypoints, goal_embedding_text,
            text, liv, image_tensors, save_path, camera, tau=args.tau)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task_path")
    parser.add_argument("episode", type=int)
    parser.add_argument("-t", "--tau", type=float, default=0.1)
    main(parser.parse_args())
