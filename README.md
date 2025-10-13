# Offline Reinforcement Learning via Inverse Optimization

This repository provides the source code for the experiments and implementation of the algorithms proposed in the paper.

## Installation

To run the provided examples, you will need to install [MOSEK](https://docs.mosek.com/10.0/install/installation.html) along with the MOSEK license. MOSEK provides free academic [licenses](https://www.mosek.com/products/academic-licenses/).

Once the MOSEK installation is completed, you can install the required packages and the research package using conda.

```bash
conda env create -f environment.yml
conda run -n io-agent python -m ipykernel install --user --name io-agent --display-name "Python (io-agent)"
```

Alternatively, you can use the `setup.sh` file to install all dependencies.

## Examples

You can find the examples under the `examples` folder:

- `examples/quadrotor.ipynb`: experiments of Section 4

The experiment directory contains Jupyter notebooks for the corresponding experiments. You can visualize the results within the notebooks.