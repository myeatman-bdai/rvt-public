import argparse
import os
import pickle

import numpy as np


def main(args):

    low_dim_obs_path = os.path.join(args.episode_path, "low_dim_obs.pkl")

    if os.path.isfile(low_dim_obs_path):
        with open(low_dim_obs_path, "rb") as f:
            to_save = []
            x = pickle.load(f)
            n = len(x._observations)
            for i in range(n):
                obs = x[i]
                to_save.append(
                    obs.joint_positions.tolist() + obs.gripper_joint_positions.tolist() + [obs.gripper_open])
            to_save = np.save(args.save_path, to_save)
    else:
        print(low_dim_obs_path, "not found")


parser = argparse.ArgumentParser()
parser.add_argument("episode_path")
parser.add_argument("save_path")
main(parser.parse_args())
