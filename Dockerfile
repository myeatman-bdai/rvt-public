FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

ENV PATH=/usr/local/cuda-11.6/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH
ENV COPPELIASIM_ROOT=/root/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
ENV LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
ENV QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
ENV PATH=$COPPELIASIM_ROOT:$PATH
ENV CUB_HOME=/root/cub-1.16.0
ENV PATH=/root/venv/bin:$PATH

ARG PATH=/usr/local/cuda-11.6/bin:$PATH
ARG LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH
ARG COPPELIASIM_ROOT=/root/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
ARG LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
ARG QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
ARG PATH=$COPPELIASIM_ROOT:$PATH
ARG CUB_HOME=/root/cub-1.16.0
ARG PATH=/root/venv/bin:$PATH

WORKDIR /root

RUN apt update -q ; apt -y install wget git build-essential wget curl tmux
RUN apt update -q ; DEBIAN_FRONTEND=noninteractive TZ="America/New_York" apt -y install tzdata

# Download CoppeliaSim.
RUN mkdir -p /shared /opt
RUN wget --no-check-certificate https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
RUN tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz && \
    rm CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz

# Download CUB.
RUN curl -LO https://github.com/NVIDIA/cub/archive/1.16.0.tar.gz
RUN tar xzf 1.16.0.tar.gz
RUN rm 1.16.0.tar.gz

# Download Python 3.8.
RUN apt update -q ; apt -y install python3.8-dev python3.8-venv
RUN wget https://bootstrap.pypa.io/get-pip.py --no-check-certificate
RUN python3 get-pip.py
RUN rm get-pip.py

# Install CoppeliaSim OpenGL deps.
RUN apt-get update -q && \
	export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y --no-install-recommends \
        vim tar xz-utils \
        libx11-6 libxcb1 libxau6 libgl1-mesa-dev \
        xvfb dbus-x11 x11-utils libxkbcommon-x11-0 \
        libavcodec-dev libavformat-dev libswscale-dev

# Create virtual env.
RUN python3.8 -m venv /root/venv

# Install pytorch3d.
RUN pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install fvcore iopath ninja
RUN FORCE_CUDA=1 pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

COPY entrypoint.sh /root/entrypoint.sh
# ENTRYPOINT [ "/root/entrypoint.sh" ]

# # Install RVT.
# RUN git clone --recurse-submodules https://github.com/NVlabs/RVT.git
# WORKDIR /root/RVT
# RUN git submodule update --init
# RUN pip install -e . 
# RUN pip install numpy cffi
# RUN pip install -e rvt/libs/PyRep 
# RUN pip install -e rvt/libs/RLBench 
# RUN pip install -e rvt/libs/YARR 
# RUN pip install -e rvt/libs/peract_colab
# WORKDIR /root

# # Fix a qt error.
# RUN pip uninstall --yes opencv-python
# RUN pip install opencv-python-headless
# # RUN for x in `conda list | grep qt` ; do conda remove --force $x ; done

# run with e.g. xvfb-run --server-args "-ac -screen 0, 1024x1024x24" python eval.py --model-folder runs/rvt  --eval-datafolder ./data/test --tasks all --eval-episodes 25 --log-name test/1 --device 0 --headless --model-name model_14.pth