import argparse
from collections import defaultdict

import numpy as np


def main(args):

    results = defaultdict(list)
    prefix = "[Evaluation] Finished"

    with open(args.path, "r") as f:

        for line in f.readlines():

            if line[:len(prefix)] == prefix:

                split = line.split(" ")
                task = split[2]
                score = float(split[6])

                results[task].append(score)

    results = dict(results)
    keys = list(sorted(results.keys()))

    for key in keys:
        mean = np.mean(results[key])
        std = np.std(results[key])
        print(f"{key}: {mean:.1f} +- {std:.1f} ({len(results[key])})")

    num_tasks = np.min([len(results[task]) for task in results.keys()])
    means = []
    for i in range(num_tasks):
        tmp = []
        for task in results.keys():
            tmp.append(results[task][i])
        means.append(np.mean(tmp))

    print()
    print(f"all tasks mean: {np.mean(means):.1f} +- {np.std(means):.1f}")
    print(means)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    main(parser.parse_args())
