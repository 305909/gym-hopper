# Domain Randomization in Robotic Control

##### by Yasaman Golshan, Sara Asadi, Francesco Giuseppe Gillio.

## Table of Contents

- [Report Abstract](#report-abstract)
- [Requirements](#requirements)
- [Environment](#environment)
- [Algorithms](#algorithms)
  - [REINFORCE](#reinforce)
  - [A2C](#a2c)
  - [SAC and ADR](#sac-and-adr)
  - [Hyperparameters Tuning](#hyperparameters-tuning)
- [Uniform Domain Randomization](#uniform-domain-randomization)
- [Automatic Domain Randomization](#automatic-domain-randomization)
- [Example of Monitoring During Training](#example-of-monitoring-during-training)
- [Footnotes](#footnotes)
- [About](#about)
- [License](#license)
- [Contributors](#contributors)

## Report Abstract

This project aims to enhance reinforcement learning (RL) agents within the Gym Hopper environment by utilizing the MuJoCo physics engine for accurate modeling. The Hopper, a one-legged robot, must learn and master jumping and maintaining balance while optimizing horizontal speed. Our approach includes implementing and comparing several RL algorithms: REINFORCE (Vanilla Policy Gradient), Actor-Critic, and Soft Actor-Critic (SAC).
To improve the agent’s performance and robustness, we introduced Uniform Domain Randomization (UDR). UDR involves varying the link masses of the Hopper robot during training, with the exception of the fixed torso mass, to expose the agent to a range of dynamic conditions. This method encourages the agent to generalize its policy across different environments, enhancing adaptability and performance.
Additionally, we implemented a curriculum learning method for Domain Randomization named AutoDR, which systematically increases the difficulty of training scenarios. Separately, we explored an Adaptive Domain Randomization method called SimOpt, which dynamically adjusts domain parameters during training. These two approaches were implemented independently to compare their effectiveness. Preliminary results indicate that combining domain randomization techniques with advanced RL algorithms significantly improves the Hopper’s stability and speed across diverse scenarios.
This work demonstrates the effectiveness of domain randomization in developing resilient robotic control strategies, contributing to the advancement of RL applications in uncertain and dynamic environments. Our findings hold the potential to inform future research and applications in robotic control and autonomous systems.

## Requirements

- [mujoco-py](https://github.com/openai/mujoco-py)
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)

#### Setup for Google Colab

Clone the repository and install the required packages by running the following commands:

`!git clone https://github.com/305909/gym-hopper.git`  
`!bash gym-hopper/setups.sh`

## Environment

The [Hopper](https://www.gymlibrary.ml/environments/mujoco/hopper/) of MuJoCo, a two-dimensional figure with one leg, comprises four primary body parts: a top torso, a middle thigh, a bottom leg, and a single foot supporting the entire body. The objective involves generating forward (rightward) movement through torque application at the three hinges connecting these body segments.
In this study, we implemented two customized versions of the Gym Hopper environment: CustomHopper-source-v0 and CustomHopper-target-v0. The main distinction between these environments concerns the mass of the Hopper's torso. Specifically, CustomHopper-source-v0 sets the torso mass at 2.53429174 units, while CustomHopper-target-v0 raises it to 3.53429174 units. 
The transition from the source to the target environment embodies the essence of sim-to-real transferability. This project aims to create algorithms capable of learning within simulated environments (source) and successfully applying acquired knowledge in real-world situations (target).

## Algorithms

The repository contains several implementations of RL algorithms, differing in control policies, learning strategies and domain randomization.

### REINFORCE

This project implements the REINFORCE (Vanilla Policy Gradient) algorithm with three variations that differ for the usage of the baseline term:
1. without baseline
2. with constant baseline
3. whitening transformation baseline

For more details, check out our custom implementation of the REINFORCE (Vanilla Policy Gradient) algorithm located in the `rein.py` file within the `agents` folder.

#### How to run the code on Google Colab

Train and test the REINFORCE (Vanilla Policy Gradient) algorithm by running the following command:

`!python /content/gym-hopper/algorithms/REF.py --train \`  
`                                              --test`

with the possibility of setting different execution parameters:

`--train: flag to start training the model`  
`--test: flag to start testing the model`  
`--render: flag to render the environment over training/testing`  
`--device: set the processing device ('cuda' for GPU, 'cpu' for CPU)`  
`--train-env: set the training environment`  
`--test-env: set the testing environment`  
`--train-episodes: set the number of training episodes`  
`--test-episodes: set the number of testing episodes`  
`--eval-frequency: set the evaluation frequency over training iterations`  
`--baseline: set the baseline for the policy update function [vanilla, constant, whitening]`  
`--input-model: set the pre-trained input model (in .mdl format)`  
`--directory: set path to the output location for checkpoint storage (model and rendering)`  

### A2C

This project implements the Advantage-Actor-Critic algorithm with a batch update method of the policy network, set to 32 time-steps per update, and two multi-layer neural networks with 3 hidden layers of 64 hidden neurons for the actor and the critic.
For more details, check out our custom implementation of the Advantage-Actor-Critic algorithm located in the `aac.py` file within the `agents` folder.

#### How to run the code on Google Colab

Train and test the A2C algorithm by running the following command:

`!python /content/gym-hopper/algorithms/A2C.py --train \`  
`                                              --test`

with the possibility of setting different execution parameters as in the previous REINFORCE algorithm.

### SAC

This project implements the Soft-Actor-Critic algorithm using the implementation of the open-source reinforcement learning library [stable-baselines3](https://github.com/DLR-RM/stable-baselines3).

#### How to run the code on Google Colab

Train and test the SAC algorithm by running the following command:

`!python /content/gym-hopper/algorithms/SAC.py --train \`  
`                                              --test`

with the possibility of setting different execution parameters as in the previous REINFORCE and A2C algorithms.

### Hyperparameters Tuning

This project also implements parameter tuning for the algorithms under investigation. The `tunings` folder contains the tuning code for each algorithm:

- `REF.py`: gridsearch algorithm for REINFORCE;
- `A2C.py`: gridsearch algorithm for A2C;
- `SAC.py`: gridsearch algorithm for SAC;

#### How to run the code on Google Colab

Search for the optimal parameter configuration for each algorithm by running the following commands:

`!python /content/gym-hopper/tunings/REF.py`  
`!python /content/gym-hopper/tunings/A2C.py`  
`!python /content/gym-hopper/tunings/SAC.py`
