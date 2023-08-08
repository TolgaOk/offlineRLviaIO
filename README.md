# From Supervised to Reinforcement Learning: an Inverse Optimization Approach

## Installation

To run the provided examples you will need to install [MOSEK](https://docs.mosek.com/10.0/install/installation.html) along with the MOSEK license. MOSEK provides free academic [license](https://www.mosek.com/products/academic-licenses/).

Once the MOSEK installation is completed you can install the required packages and the research package.

```bash
pip install -r requirements.txt
pip install -e .
```

This repository contains several experiments that contains comparison between IO agent and [CQL (Conservative Q-Learning)](https://arxiv.org/abs/2006.04779). Additionally, it includes comparison with MPC agents and [PPO (Proximal Policy Optimization) algorithm](https://arxiv.org/abs/1707.06347). These experiments are run on Quadrotor environment provided in [safe-control-gym](https://arxiv.org/abs/2108.06266). In order to run these experiments, an additional installation process is required.

These steps can be done by following the installation process of the listed repositories below.
- [safe-control-gym](https://github.com/utiasDSL/safe-control-gym) for the Quadrotor environment.
- [OfflineRL-Kit](https://github.com/yihaosun1124/OfflineRL-Kit/tree/main) for the CQL agent.
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/guide/install.html) for the PPO agent.

- - -
## Examples

You can run the example notebook:

- `examples/fighter/*` : experiments of Sections 4

The experiment directory contains the jupyter-notebook for the corresponding experiment. You can visualize the results within the notebooks.