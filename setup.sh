#!/bin/bash

set -e

python3 -m venv venv --system-site-packages --symlinks
source venv/bin/activate

pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116

sudo apt install gcc-10 g++-10
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
sudo ln -s /usr/bin/gcc-10 /usr/local/cuda-11.6/bin/gcc
sudo ln -s /usr/bin/g++-10 /usr/local/cuda-11.6/bin/g++

pip install ninja
FORCE_CUDA=1 pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

source env_sim.sh
pip install -e . 
pip install -e rvt/libs/PyRep 
pip install -e rvt/libs/RLBench 
pip install -e rvt/libs/YARR 
pip install -e rvt/libs/peract_colab

pip install open3d

# ROS dependencies.
pip install empy lark