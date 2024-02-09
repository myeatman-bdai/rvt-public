python liv_rlbench_curves_v1.py data/test/close_jar 0
python liv_rlbench_curves_v1.py data/test/open_drawer 0
python liv_rlbench_curves_v1.py data/test/place_shape_in_shape_sorter 0
python liv_rlbench_curves_v1.py data/test/stack_blocks 0

python liv_rlbench_curves_v1.py data/test/close_jar 0 -s 230929_finetune_rlbench_subset_front/snapshot_10000.pt
python liv_rlbench_curves_v1.py data/test/open_drawer 0 -s 230929_finetune_rlbench_subset_front/snapshot_10000.pt
python liv_rlbench_curves_v1.py data/test/place_shape_in_shape_sorter 0 -s 230929_finetune_rlbench_subset_front/snapshot_10000.pt
python liv_rlbench_curves_v1.py data/test/stack_blocks 0 -s 230929_finetune_rlbench_subset_front/snapshot_10000.pt

python liv_rlbench_curves_v1.py data/test/close_jar 0 -s 230929_finetune_rlbench_subset_wrist/snapshot_10000.pt
python liv_rlbench_curves_v1.py data/test/open_drawer 0 -s 230929_finetune_rlbench_subset_wrist/snapshot_10000.pt
python liv_rlbench_curves_v1.py data/test/place_shape_in_shape_sorter 0 -s 230929_finetune_rlbench_subset_wrist/snapshot_10000.pt
python liv_rlbench_curves_v1.py data/test/stack_blocks 0 -s 230929_finetune_rlbench_subset_wrist/snapshot_10000.pt
