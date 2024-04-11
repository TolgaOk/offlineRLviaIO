# Offline Reinforcement Learning via Inverse Optimization

This repository provides the source code of the  experiments and implementation of the algorithms proposed in the paper.

## Installation

To run the provided examples you will need to install [MOSEK](https://docs.mosek.com/10.0/install/installation.html) along with the MOSEK license. MOSEK provides free academic [license](https://www.mosek.com/products/academic-licenses/).

Once the MOSEK installation is completed you can install the required packages and the research package.

```bash
pip install -r requirements.txt
pip install -e .
```

Alternatively, you can use [apptainer](https://apptainer.org/) to build a self contained image using the ```image.def``` file. Run ```start.sh --build``` to build a apptainer image and run ```start.sh --run``` to start a container running vs-code server.

### Additional packages

This repository contains several experiments that contains comparison between IO agent and several other RL algorithms. These experiments are run on Quadrotor environment provided in [safe-control-gym](https://arxiv.org/abs/2108.06266) and [MuJoCo](https://mujoco.org/) control benchmark. In order to run these experiments, an additional installation process is required.

These steps can be done by following the installation process of the listed repositories below.

- [safe-control-gym](https://github.com/utiasDSL/safe-control-gym) for the Quadrotor environment.
- [OfflineRL-Kit](https://github.com/yihaosun1124/OfflineRL-Kit/tree/main) for the offline RL agent.
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/guide/install.html) for the PPO agent.
- [D4RL](https://github.com/Farama-Foundation/D4RL) for offline MuJoCo control benchmark datasets.
- For the iterative IO agent:
    - [jax](https://github.com/google/jax)
    - [jaxopt](https://github.com/google/jaxopt) 
    - [jaxtyping](https://github.com/patrick-kidger/jaxtyping)
    - [optax](https://github.com/google-deepmind/optax)

- - -
## Examples

You can find the examples under the ```examples``` folder:

- `examples/quadrotor.ipynb` : experiments of Sections 4

The experiment directory contains jupyter-notebooks for the corresponding experiments. You can visualize the results within the notebooks.