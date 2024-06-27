#!/bin/bash

# update and install Hopper environment system requirements
apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    patchelf

# install python packages
pip install gym
pip install free-mujoco-py
pip install "stable-baselines3[extra]>=2.0.0a4"
