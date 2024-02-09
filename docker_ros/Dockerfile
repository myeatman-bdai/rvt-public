FROM ubuntu:jammy

ENV PATH="/root/miniconda3/bin:${PATH}"
ENV COPPELIASIM_ROOT=/root/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
ENV LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
ENV QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
ENV PATH=$COPPELIASIM_ROOT:$PATH
ENV LANG=en_US.UTF-8

ARG PATH="/root/miniconda3/bin:${PATH}"
ARG COPPELIASIM_ROOT=/root/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
ARG LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
ARG QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
ARG PATH=$COPPELIASIM_ROOT:$PATH
ARG LANG=en_US.UTF-8

WORKDIR /root

RUN apt update ; apt -y install wget git build-essential
RUN apt update ; DEBIAN_FRONTEND=noninteractive TZ="America/New_York" apt -y install tzdata

# Install CoppeliaSim.
RUN apt-get update -q && \
	export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y --no-install-recommends \
        vim tar xz-utils \
        libx11-6 libxcb1 libxau6 libgl1-mesa-dev \
        xvfb dbus-x11 x11-utils libxkbcommon-x11-0 \
        libavcodec-dev libavformat-dev libswscale-dev \
        && \
    apt-get autoclean -y && apt-get autoremove -y && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt update ; apt -y install wget

RUN mkdir -p /shared /opt

RUN wget --no-check-certificate https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
RUN tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz && \
    rm CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz

# Miniconda.
RUN wget --no-check-certificate \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda init bash
RUN conda create -n rvt python=3.9
SHELL ["conda", "run", "-n", "rvt", "/bin/bash", "-c"]

# Install pytorch3d.
RUN conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
RUN conda install -c fvcore -c iopath -c conda-forge fvcore iopath
RUN conda install -c bottler nvidiacub
RUN conda install pytorch3d -c pytorch3d

# Install RVT.
RUN git clone --recurse-submodules https://github.com/NVlabs/RVT.git
WORKDIR /root/RVT
RUN git submodule update --init
RUN pip install -e . 
RUN pip install numpy cffi
RUN pip install -e rvt/libs/PyRep 
RUN pip install -e rvt/libs/RLBench 
RUN pip install -e rvt/libs/YARR 
RUN pip install -e rvt/libs/peract_colab
WORKDIR /root

# Fix a qt error.
RUN pip uninstall --yes opencv-python
RUN pip install opencv-python-headless
RUN for x in `conda list | grep qt` ; do conda remove --force $x ; done

# Install ROS 2 Humble.
RUN apt update && apt -y install locales
RUN locale-gen en_US en_US.UTF-8
RUN update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

RUN apt update ; apt -y install software-properties-common
RUN add-apt-repository universe

RUN apt update && apt -y install curl
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

RUN apt update ; apt install -y ros-humble-desktop ros-dev-tools

RUN echo "\n /opt/ros/humble/setup.bash \n" >> /root/.bashrc

# run with e.g. xvfb-run --server-args "-ac -screen 0, 1024x1024x24" python eval.py --model-folder runs/rvt  --eval-datafolder ./data/test --tasks all --eval-episodes 25 --log-name test/1 --device 0 --headless --model-name model_14.pth