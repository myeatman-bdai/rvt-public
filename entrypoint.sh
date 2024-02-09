#!/bin/bash

# Install pytorch3d.
# FORCE_CUDA=1 pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

source /root/venv/bin/activate

# Install RVT.
cd /root/rvt
pip install -e . 
pip install numpy cffi
pip install -e rvt/libs/PyRep 
pip install -e rvt/libs/RLBench 
pip install -e rvt/libs/YARR 
pip install -e rvt/libs/peract_colab
cd /root

# Fix a qt error.
pip uninstall --yes opencv-python
pip install opencv-python-headless
# RUN for x in `conda list | grep qt` ; do conda remove --force $x ; done

# Fix np.bool error.
pip install numpy==1.20

# Fix tensorboard error.
pip install tensorflow==2.9.1

bash