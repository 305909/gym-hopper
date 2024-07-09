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

In this project, we focus on enhancing reinforcement learning (RL) agents for the Gym Hopper environment, utilizing the MuJoCo physics engine for accurate modeling. The Hopper, a one-legged robot, must learn to jump and maintain balance while optimizing its horizontal speed. Our approach includes implementing and comparing several RL algorithms: REINFORCE (Vanilla Policy Gradient), Actor-Critic, and Soft Actor-Critic (SAC).
To improve the agent’s performance and robustness, we introduced Uniform Domain Randomization (UDR). UDR involves varying the link masses of the Hopper robot during training, with the exception of the fixed torso mass, to expose the agent to a range of dynamic conditions. This method encourages the agent to generalize its policy across different environments, enhancing adaptability and performance.
Additionally, we implemented a curriculum learning method for Domain Randomization named AutoDR, which systematically increases the difficulty of training scenarios. Separately, we are exploring an Adaptive Domain Randomization method called SimOpt, which dynamically adjusts domain parameters during training. These two approaches are implemented independently to compare their effectiveness. Preliminary results indicate that combining domain randomization techniques with advanced RL algorithms significantly improves the Hopper’s stability and speed across diverse scenarios.
This work demonstrates the effectiveness of domain randomization in developing resilient robotic control strategies, contributing to the advancement of RL applications in uncertain and dynamic environments. Our findings have the potential to inform future research and applications in robotic control and autonomous systems.

## Requirements

- Mujoco-py
- stable-baselines3
