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
Additionally, we implemented a curriculum learning method for Domain Randomization named ADR, which systematically increases the difficulty of training scenarios. Separately, we explored an Adaptive Domain Randomization method called SimOpt, which dynamically adjusts domain parameters during training. These two approaches were implemented independently to compare their effectiveness. Preliminary results indicate that combining domain randomization techniques with advanced RL algorithms significantly improves the Hopper’s stability and speed across diverse scenarios.
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
- `--input-model`: set the pre-trained input model (in .mdl format)
- `--directory`: set path to the output location for checkpoint storage (model and rendering)`  

### A2C

This project implements the A2C (Advantage-Actor-Critic) algorithm with two variations that differ for the update method of the policy network:
1. `stepwise` fashion
2. `batch` fashion

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

This project implements a `CustomHopper-source-UDR-v0` custom environment to introduce Uniform Domain Randomization (UDR). UDR involves varying the link masses of the Hopper robot during training, while maintaining the torso mass constant, to expose the agent to a range of dynamic conditions. For each link $i$, the environment instantiates the boundaries of the physical parameter distribution ($\mathbb{U_\phi}$) and randomly samples the $i$-th link mass at the beginning of each episode:

$$
m_i \sim \mathbb{U_\phi}((1 - \phi) \cdot m_{i_0}, (1 + \phi) \cdot m_{i_0})
$$

where:
- $\mathit{m_{i_0}} \rightarrow$ the original mass of the $i$-th link of the Hopper robot;
- $\mathit{\phi = 0.5} \rightarrow$ the variation factor;
- $\mathbb{U_\phi}(a, b) \rightarrow$ continuous uniform distribution between $\mathit{a}$ and $\mathit{b}$ with variation factor $\phi$.
  
For more details, check out our custom implementation of the `CustomHopper-source-UDR-v0` environment in the `custom_hopper.py` file inside the `env` folder.

#### How to run the code on Google Colab

To enable Uniform Domain Randomization, set the custom environment `CustomHopper-source-UDR-v0` as testing environment, i.e. the execution parameter `test_env` to `'source-UDR'`. Train and test the SAC algorithm with UDR by running the following command:

```python
# run SAC (Soft-Actor-Critic) algorithm with UDR
!python /content/gym-hopper/algorithms/SAC.py --train \
                                              --test \
                                              --train-env 'source-UDR'
```

## Control Domain Randomization

Control Domain Randomization (CDR) controls the domain randomization process. CDR involves dynamically varying the link masses of the Hopper robot during training, while maintaining the torso mass constant, to enhance the adaptability of the agent across varying scenarios. The algorithm systematically adjusts the variation factor $\phi$ of the physical parameter distribution based on the agent's performance, thus facilitating optimal management of exploration and exploitation in different environmental contexts.

### Initialization and Domain Configuration

Upon initialization, the ADR module initializes the following parameters:

- $\mathit{\phi^m = 3.5} \rightarrow$ upper bound for the variation factor, defining the range of parameter adjustments;
- $\mathit{\delta = 0.05} \rightarrow$ step size for updating the variation factor $\phi^k$ based on performance feedback;
- $\mathit{\phi^0 = 0.0} \rightarrow$ initial variation factor applied to the physical parameters;
- $\mathit{\alpha} \rightarrow$ threshold rate to update physical parameters;
- $\mathit{D^{L}, D^{H}} \rightarrow$ data buffers storing the lower and upper performance thresholds for parameter adjustment.

Within the CDR framework, $\mathit{D^{L}}$ and $\mathit{D^{H}}$ represent the thresholds coming from the performance metrics of two benchmark policies:

- simulation policy $\pi_{s}$: trained in the `source` environment (simulation) without domain randomization;
- real-world policy $\pi_{r}$: trained in the `target` environment (real world).

### Domain Randomization

For each link $i$, the environment randomly samples the $i$-th link mass at the beginning of each episode according to the current variation factor $\phi^k$ and the probability distribution: 

$$
m_i \sim \mathbb{U_{\phi^k}}((1 - \phi^k) \cdot m_{i_0}, (1 + \phi^k) \cdot m_{i_0})
$$

$$
m_i \sim \mathbb{N_{\phi^k}}(m_{i_0}, \phi^k)
$$

where:
- $\mathit{m_{i_0}} \rightarrow$ the original mass of the $i$-th link of the Hopper robot;
- $\mathit{\phi^k} \rightarrow$ the current variation factor;
- $\mathbb{U_{\phi^k}}(a, b) \rightarrow$ continuous uniform distribution between $\mathit{a}$ and $\mathit{b}$ with variation factor $\phi^k$;
- $\mathbb{N_{\phi^k}}(\mu, \sigma^2) \rightarrow$ continuous normal distribution with mean $\mathit{\mu}$ and standard deviation $\mathit{\sigma^2}$ with variation factor $\phi^k$.

### Performance Evaluation and $\phi$ Update:

CDR pauses the training process every $K$ number of episodes and iterates over $N$ testing episodes to evaluate the agent's performance (shift to the `target` environment). The algorithm then updates the current variation factor $\phi^k$ based on agent's performance $r$, i.e. the reward rate within the optimization interval:

$$
r = \frac{1}{N} \sum_{n=1}^{N} [ 1 | if D_k^{L} < G_{n}^{\pi} < D_k^{H} ]
$$

$$
r = \sum_{G_{n=1}^{\pi} \in G_{N}} 1 \text{ if } D_k^{L} \leq Gt \leq D_k^{H}
$$

$$
\phi^{k+1} = \begin{cases} 
\phi^k - \delta & \text{if } \mathbb{E}[G^\pi] > D_k^{H} \\
\phi^k + \delta & \text{if } D_k^{L} \leq \mathbb{E}[G^\pi] \leq D_k^{H} \\
\phi^k & \text{otherwise}
\end{cases}
$$

where:

$$
D_k^{L} = \mathbb{E}[G^{\pi_s}] = \frac{1}{N} \sum_{n=1}^{N} G_{n}^{\pi_s}
$$  

$$
D_k^{H} = \mathbb{E}[G^{\pi_r}] = \frac{1}{N} \sum_{n=1}^{N} G_{n}^{\pi_r}
$$

#### How to run the code on Google Colab

Train and test the SAC algorithm with ADR by running the following command:

```python
# run SAC (Soft-Actor-Critic) algorithm
!python /content/gym-hopper/algorithms/ADR.py --train \
                                              --test
```

The `ADR.py` code offers the chance to set several execution parameters as in the previous code for the `SAC` algorithm.
