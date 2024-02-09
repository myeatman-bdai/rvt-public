import argparse
import os
import pickle


def main(args):

    low_dim_obs_path = os.path.join(args.episode_path, "low_dim_obs.pkl")
    variation_descriptions_path = os.path.join(args.episode_path, "variation_descriptions.pkl")
    variation_number_path = os.path.join(args.episode_path, "variation_number.pkl")

    if os.path.isfile(low_dim_obs_path):
        with open(low_dim_obs_path, "rb") as f:
            x = pickle.load(f)
            print("low_dim_obs len:", len(x._observations))
            obs = x[0]
            d = obs.__dict__
            for key, value in d.items():
                print(key, value is None)
            print("joint positions:", d["joint_positions"])
    else:
        print(low_dim_obs_path, "not found")

    if os.path.isfile(variation_descriptions_path):
        with open(variation_descriptions_path, "rb") as f:
            x = pickle.load(f)
            print("variation_descriptions:", x)
    else:
        print(variation_descriptions_path, "not found")

    if os.path.isfile(variation_number_path):
        with open(variation_number_path, "rb") as f:
            x = pickle.load(f)
            print("variation_number:", x)
    else:
        print(variation_number_path, "not found")



parser = argparse.ArgumentParser()
parser.add_argument("episode_path")
main(parser.parse_args())
