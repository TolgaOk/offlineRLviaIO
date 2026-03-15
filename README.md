# Offline Reinforcement Learning via Inverse Optimization

[![arXiv](https://img.shields.io/badge/arXiv-2502.20030-b31b1b.svg)](https://arxiv.org/abs/2502.20030)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

Official implementation of the paper "Offline Reinforcement Learning via Inverse Optimization".

## Installation

This project uses [MOSEK](https://www.mosek.com/) for optimization. You will need a MOSEK [license](https://www.mosek.com/products/academic-licenses/) (free for academic use).

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate io-agent

# Install the package in editable mode
pip install -e .
```

### Dependencies

The conda environment includes all dependencies needed for the experiments. The following external packages are installed automatically via `environment.yml`:

- [safe-control-gym](https://github.com/utiasDSL/safe-control-gym) for the Quadrotor environment.
- [D4RL](https://github.com/Farama-Foundation/D4RL) for offline MuJoCo control benchmark datasets.
- [OfflineRL-Kit](https://github.com/yihaosun1124/OfflineRL-Kit/tree/main) for the offline RL agents.
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/guide/install.html) for the PPO agent.
- [JAX](https://github.com/google/jax), [jaxopt](https://github.com/google/jaxopt), [jaxtyping](https://github.com/patrick-kidger/jaxtyping), [optax](https://github.com/google-deepmind/optax) for the iterative IO agent.

## Experiments

You can find the examples under the `examples` folder:

- `examples/quadrotor.ipynb`: Quadrotor experiments
- `examples/dualheater.ipynb`: Dual heater experiments
- `examples/fighter.ipynb`: Fighter experiments
- `examples/mujoco/trainer.py`: run offline RL algorithms (see `--help` for details)
- `examples/mujoco/fast_io_trainer.py`: run iterative IO algorithm (see `--help` for details)

## Citation

If you find this work useful, please cite:

```bibtex
@article{Dimanidis2025,
  title={Offline Reinforcement Learning via Inverse Optimization},
  author={Dimanidis, Ioannis and Ok, Tolga and Esfahani, Peyman Mohajerin},
  journal={arXiv preprint arXiv:2502.20030},
  year={2025}
}
```
