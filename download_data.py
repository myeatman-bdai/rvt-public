import os
import subprocess
from multiprocessing import Pool


IDS = [
"1qkJA4uCMbWJ7TY0VbfBgOxLwCP-K5WC9",
"1rZL7zaExCql0z9uTqsDrp4kDm4-26tdU",
"1msB0N5O8W4KiiecgUi0zEMlUFJLqDPFF",
"1yHZKvhk9mn6bJWSP0RD-rX0Mupox3HKx",
"1jY1WB5pVVB_DLumO7z-8g9Kng6XYw5UP",
"1xwL17PD3pFUNcU7tDzP-zeoVoXTvtFEj",
"1yCtIbB37nnFAGARl1rJKJrV8bZZ0JdZg",
"1cT1J0QS11qELaTBMQ6dvJYDhnxhecDWR",
"1X4Nsx-pMYDWzo9EdGfdosAiZUTJZSPll",
"1TwywrMqNYpDmXxeBPjZYodir_zN390rW",
"1VDw02NdwnQb9e111uS8c-IEEwo0e9ikl",
"1hvkRJHlIszWFLTcbkPijZ8DuEK1-uaqm",
"1TmQJKvQd_lEWUbfOnHaq1InxOZgK40sY",
"1lrCglCFovEVqlQDHen49atYufM4A9dk7",
"1827-OByTnW_aeaHRzWp5j87BL0F3f4Bu",
"1b0mPNEpPkN9HfY1XDTMPBMfYdJzkpk2X",
"1gdn-tPV3stJt9fpxBA4CMAnIijGjV_Il",
"19uGLa4NeZPPNtEY0s2rbRvBkU0lsgCrw"
]
    
NAMES = [
    "turn_tap.zip",
    "sweep_to_dustpan_of_size.zip",
    "stack_cups.zip",
    "stack_blocks.zip",
    "slide_block_to_color_target.zip",
    "reach_and_drag.zip",
    "put_money_in_safe.zip",
    "put_item_in_drawer.zip",
    "put_groceries_in_cupboard.zip",
    "push_buttons.zip",
    "place_wine_at_rack_location.zip",
    "place_shape_in_shape_sorter.zip",
    "place_cups.zip",
    "open_drawer.zip",
    "meat_off_grill.zip",
    "light_bulb_in.zip",
    "insert_onto_square_peg.zip",
    "close_jar.zip"
]

assert len(IDS) == len(NAMES)

ROOT = "rvt/data/train"


def worker(x):

    name, id = x

    zip_path = os.path.join(ROOT, name)
    dir_path = os.path.join(ROOT, ".".join(name.split(".")[:-1]))
    
    if not os.path.isdir(dir_path):
        # subprocess.call(f"gdown --no-cookies {id} -O {zip_path}", shell=True)
        subprocess.call(f"unzip {zip_path} -d {ROOT}", shell=True)
        # subprocess.call(f"rm {zip_path}", shell=True)


def main():

    os.makedirs(ROOT, exist_ok=True)

    pool_args = []
    for name, id in zip(NAMES, IDS):
        pool_args.append((name, id))

    pool = Pool(processes=4)
    pool.map(worker, pool_args)
    pool.join()


if __name__ == "__main__":
    main()
