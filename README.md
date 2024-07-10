# Domain Randomization in Robotic Control

##### by Yasaman Golshan, Sara Asadi, Francesco Giuseppe Gillio.

## Table of Contents

- [Report Abstract](#report-abstract)
- [Requirements](#requirements)
- [Environment](#environment)
- [Algorithms](#algorithms)
  - [REINFORCE](#reinforce)
  - [A2C](#a2c)
  - [SAC](#sac)
  - [Hyperparameters Tuning](#hyperparameters-tuning)
- [Uniform Domain Randomization](#uniform-domain-randomization)
- [Automatic Domain Randomization](#automatic-domain-randomization)
- [Example of Training](#example-of-monitoring-during-training)
- [About](#about)
- [License](#license)
- [Contributors](#contributors)

## Report Abstract

This project aims to enhance reinforcement learning (RL) agents within the Gym Hopper environment by utilizing the MuJoCo physics engine for accurate modeling. The Hopper, a one-legged robot, must learn and master jumping and maintaining balance while optimizing horizontal speed. Our approach includes implementing and comparing several RL algorithms: REINFORCE (Vanilla Policy Gradient), Advantage-Actor-Critic (A2C), and Soft Actor-Critic (SAC).
To improve the agent’s performance and robustness, we introduced Uniform Domain Randomization (UDR). UDR involves varying the link masses of the Hopper robot during training, with the exception of the fixed torso mass, to expose the agent to a range of dynamic conditions. This method encourages the agent to generalize its policy across different environments, enhancing adaptability and performance.
Additionally, we implemented a curriculum learning method for Domain Randomization named AutoDR, which systematically increases the difficulty of training scenarios. Separately, we explored an Adaptive Domain Randomization method called SimOpt, which dynamically adjusts domain parameters during training. These two approaches were implemented independently to compare their effectiveness. Preliminary results indicate that combining domain randomization techniques with advanced RL algorithms significantly improves the Hopper’s stability and speed across diverse scenarios.
This work demonstrates the effectiveness of domain randomization in developing resilient robotic control strategies, contributing to the advancement of RL applications in uncertain and dynamic environments. Our findings hold the potential to inform future research and applications in robotic control and autonomous systems.

## Requirements

- [mujoco-py](https://github.com/openai/mujoco-py)
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)

#### Setup for Google Colab

Clone the repository and install the Hopper environment system requirements packages by running the following commands:

```python
!git clone https://github.com/305909/gym-hopper.git
!bash gym-hopper/setups.sh
```

## Environment

The [Hopper](https://www.gymlibrary.ml/environments/mujoco/hopper/) of MuJoCo, a two-dimensional figure with one leg, comprises four primary body parts: a top torso, a middle thigh, a bottom leg, and a single foot supporting the entire body. The objective involves generating forward (rightward) movement through torque application at the three hinges connecting these body segments.
In this study, we implemented two customized versions of the Gym Hopper environment: `CustomHopper-source-v0` and `CustomHopper-target-v0`. The main distinction between these environments concerns the mass of the Hopper's torso. Specifically, `CustomHopper-source-v0` sets the torso mass at `2.53429174` kg, while `CustomHopper-target-v0` raises it to `3.53429174` kg. 
The transition from the source to the target environment embodies the essence of sim-to-real transferability. This project aims to create algorithms capable of learning within simulated environments (source) and successfully applying acquired knowledge in real-world situations (target).

## Algorithms

The repository contains several implementations of RL algorithms, differing in control policies, learning strategies and domain randomization.

### REINFORCE

This project implements the REINFORCE (Vanilla Policy Gradient) algorithm with three variations that differ for the usage of the baseline term:
1. without baseline
2. with constant baseline `baseline = 20`
3. whitening transformation baseline

For more details, check out our custom implementation of the REINFORCE (Vanilla Policy Gradient) algorithm in the `rein.py` file inside the `agents` folder.

#### How to run the code on Google Colab

Train and test the REINFORCE (Vanilla Policy Gradient) algorithm by running the following command:

```python
# run REINFORCE (Vanilla Policy Gradient) algorithm
!python /content/gym-hopper/algorithms/REF.py --train \
                                              --test
```

The `REF.py` code offers the chance to set several execution parameters:

- `--train`: flag to start training the model
- `--test`: flag to start testing the model
- `--render`: flag to render the environment over training/testing
- `--device`: set the processing device ('cuda' for GPU, 'cpu' for CPU)
- `--train-env`: set the training environment
- `--test-env`: set the testing environment
- `--train-episodes`: set the number of training episodes
- `--test-episodes`: set the number of testing episodes
- `--eval-frequency`: set the evaluation frequency over training iterations
- `--baseline`: set the baseline for the policy update function (vanilla, constant, whitening)
- `--input-model`: set the pre-trained input model (in .mdl format)
- `--directory`: set path to the output location for checkpoint storage (model and rendering)`  

### A2C

This project implements the Advantage-Actor-Critic algorithm with a batch update method of the policy network, set to `32` time-steps per update, and two multi-layer neural networks with 3 hidden layers for the actor and the critic.
For more details, check out our custom implementation of the Advantage-Actor-Critic algorithm in the `aac.py` file inside the `agents` folder.

#### How to run the code on Google Colab

Train and test the A2C algorithm by running the following command:

```python
# run A2C (Advantage-Actor-Critic) algorithm
!python /content/gym-hopper/algorithms/A2C.py --train \
                                              --test
```

The `A2C.py` code offers the chance to set several execution parameters as in the previous code for the `REINFORCE` algorithm.

### SAC

This project implements the Soft-Actor-Critic algorithm using the open-source reinforcement learning library [stable-baselines3](https://github.com/DLR-RM/stable-baselines3).

#### How to run the code on Google Colab

Train and test the SAC algorithm by running the following command:

```python
# run SAC (Soft-Actor-Critic) algorithm
!python /content/gym-hopper/algorithms/SAC.py --train \
                                              --test
```

The `SAC.py` code offers the chance to set several execution parameters as in the previous codes for the `REINFORCE` and `A2C` algorithms.

### Hyperparameters Tuning

This project also implements parameter tuning for the algorithms under investigation. The `tunings` folder contains the tuning code for each algorithm:

- `REF.py`: code to tune the `REINFORCE` parameters;
- `A2C.py`: code to tune the `A2C` parameters;
- `SAC.py`: code to tune the `SAC` parameters;

#### How to run the code on Google Colab

Search for the optimal parameter configuration for each algorithm by running the following commands:

```python
# run gridsearch algorithm for the REINFORCE model
!python /content/gym-hopper/tunings/REF.py
```
```python
# run gridsearch algorithm for the A2C model
!python /content/gym-hopper/tunings/A2C.py`
```
```python
# run gridsearch algorithm for the SAC model
!python /content/gym-hopper/tunings/SAC.py
```

## Uniform Domain Randomization

This project implements a `CustomHopper-source-UDR-v0` custom environment to introduce Uniform Domain Randomization (UDR). UDR involves varying the link masses of the Hopper robot during training, with the exception of the fixed torso mass, to expose the agent to a range of dynamic conditions. For each mass separately, the environment instantiates the boundaries of the parameter distribution and randomly samples parameters at the beginning of each episode:  

$$
m_i \sim \mathbb{U}((1 - \phi) \cdot m_{i_0}, (1 + \phi) \cdot m_{i_0})
$$

where:
- $\mathit{m_{i_0}} \rightarrow$ the original mass of the $i$-th link of the Hopper robot,
- $\mathit{\phi = 0.25} \rightarrow$ the variation factor,
- $\mathbb{U}(a, b) \rightarrow$ a continuous uniform distribution between $\mathit{a}$ and $\mathit{b}$.
  
For more details, check out our custom implementation of the `CustomHopper-source-UDR-v0` environment in the `custom_hopper.py` file inside the `env` folder.

#### How to run the code on Google Colab

To enable Uniform Domain Randomization, set the custom environment `CustomHopper-source-UDR-v0` as testing environment, i.e. the execution parameter `test_env` to `'source-UDR'`. Train and test the SAC algorithm with UDR by running the following command:

```python
# run SAC (Soft-Actor-Critic) algorithm with UDR
!python /content/gym-hopper/algorithms/SAC.py --train \
                                              --test \
                                              --train-env 'source-UDR'
```

## Automatic Domain Randomization

Automatic Domain Randomization (ADR) automates the domain randomization process. ADR involves dynamically varying the link masses of the Hopper robot during training, with the exception of the fixed torso mass. The algorithm systematically adjusts the randomization parameters according to the agent's performance, thereby facilitating optimal management of exploration and exploitation across diverse environmental settings.

### Initialization and Domain Configuration

At initialization the environment sets the ADR parameters:

- $\mathit{\phi^m = 2.0} \rightarrow$ upper bound for the variation factor,
- $\mathit{\delta = 0.05} \rightarrow$ step size for updating the variation factor,
- $\mathit{\phi^0 = 0.1} \rightarrow$ initial variation factor,
- $\mathit{{D^{L}, D^{H}}} \rightarrow$ performance data buffers storing the lower and upper performance bounds for each episode.

### Domain Randomization

For each mass separately, the environment randomly samples parameters at the beginning of each episode according to the the current variation factor $\phi^e$:  

$$
m_i \sim \mathbb{U}((1 - \phi^e) \cdot m_{i_0}, (1 + \phi^e) \cdot m_{i_0})
$$

where:
- $\mathit{m_{i_0}} \rightarrow$ the original mass of the $i$-th link of the Hopper robot,
- $\mathit{\phi^e} \rightarrow$ the current variation factor,
- $\mathbb{U}(a, b) \rightarrow$ a continuous uniform distribution between $\mathit{a}$ and $\mathit{b}$.

### Performance Evaluation and $\phi$ Update:

ADR pauses the training process every $M$ number of episodes and iterates over $N$ testing episodes to evaluate the agent's performance (shifting the environment). The algorithm then updates the $\phi$ variation factor according to the agent's performance $\bar{G}$, i.e. the average cumulative reward over the $N$ testing episodes:

$$
\bar{G} = \frac{1}{N} \sum_{e=1}^{N}G_{T_e}
$$

$$
\phi^{e+1} = \begin{cases} 
\phi^e - \delta & \text{if } \bar{G} > D_e^{H} \\
\phi^e + \delta & \text{if } D_e^{L} \leq \bar{G} \leq D_e^{H} \\
\phi^e & \text{otherwise}
\end{cases}
$$

where:
- $\mathit{\phi^{e+1}} \rightarrow$ the updated value of $\phi$,
- $\mathit{\phi^e} \rightarrow$ the current variation factor,

The thresholds $\mathit{{D^{L}, D^{H}}}$ determine whether $\phi^{e+1}$ increases, decreases, or remains unchanged.

#### How to run the code on Google Colab

Train and test the SAC algorithm with ADR by running the following command:

```python
# run SAC (Soft-Actor-Critic) algorithm
!python /content/gym-hopper/algorithms/ADR.py --train \
                                              --test
```

The `ADR.py` code offers the chance to set several execution parameters as in the previous code for the `SAC` algorithm.
