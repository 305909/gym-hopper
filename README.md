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

To clone the repository and install the requirements, follow these steps:

`!git clone https://github.com/305909/gym-hopper.git`
`!bash gym-hopper/setups.sh`

## Environment

The [Hopper](https://www.gymlibrary.ml/environments/mujoco/hopper/) of MuJoCo, a two-dimensional figure with one leg, comprises four primary body parts: a top torso, a middle thigh, a bottom leg, and a single foot supporting the entire body. The objective involves generating forward (rightward) movement through torque application at the three hinges connecting these body segments.
In this study, we implemented two customized versions of the Gym Hopper environment: CustomHopper-source-v0 and CustomHopper-target-v0. The main distinction between these environments concerns the mass of the Hopper's torso. Specifically, CustomHopper-source-v0 sets the torso mass at 2.53429174 units, while CustomHopper-target-v0 raises it to 3.53429174 units. 
The transition from the source to the target environment embodies the essence of sim-to-real transferability. This project aims to create algorithms capable of learning within simulated environments (source) and successfully applying acquired knowledge in real-world situations (target).

## Algorithms

The repository contains several implementations of RL algorithms, differing in control policies, learning strategies and domain randomization.

### REINFORCE

Three variations of the REINFORCE (Vanilla Policy Gradient) algorithm. The implementations differ for the usage of the baseline term:
1. without baseline
2. with constant baseline
3. whitening transformation baseline

#### How to run the code

Running `REF.py` inside the `algorithms` folder will start a training by episodes on the specifiable environment, with the possibility to:

- set the training environment;
- set the testing environment;
- set the number of training episodes;
- set the number of testing episodes;
- set the evaluation frequency over training iterations;
- resume training from a previous model;

It is recommended to run the file with '--help' to list all available options.
