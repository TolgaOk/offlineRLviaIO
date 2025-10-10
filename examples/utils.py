from functools import partial
import numpy as np
import multiprocessing

from io_agent.plant.base import OldSchoolWrapper
import torch
from collections import defaultdict
from itertools import product, chain
import os
import json
from typing import Tuple, NamedTuple, List
from io_agent.plant.quadrotor import QuadrotorEnv
from io_agent.runner.basic import run_mpc
from io_agent.utils import parallelize, save_experiment, load_experiment
from io_agent.plant.base import OldSchoolWrapper
from io_agent.evaluator import Transition
from dataclasses import asdict


def generate_mpc_quadrotor_dataset(horizon: int = 25,
                                   n_dataset_trials: int = 100,
                                   seed: int = 42
                                   ) -> None:
    n_cpu = multiprocessing.cpu_count()
    seed_rng = np.random.default_rng(seed)

    plant = QuadrotorEnv()
    dataset_trial_seeds = seed_rng.integers(0, 2**30, n_dataset_trials)

    dataset_trajectories = parallelize(
        n_proc=min(n_cpu, n_dataset_trials),
        fn=partial(run_mpc, plant=plant),
        kwargs_list=[
            dict(
                horizon=horizon,
                use_foresight=False,  # Without hindsight data
                bias_aware=False,
                env_reset_rng=np.random.default_rng(_seed)
            ) for _seed in dataset_trial_seeds
        ],
        loading_bar_kwargs=dict(desc="MPC dataset trials")
    )

    save_experiment(
        values={"mpc_trajectories": dataset_trajectories},
        seed=seed,
        exp_dir="./quadrotor_data/dataset",
        name="mpc_without_hindsight"
    )

def trajectory_to_dataset(trajectory: List[Transition], env: QuadrotorEnv) -> dict:
    dict_dataset = {
        name: np.stack([asdict(transition)[key]
                       for transition in chain(*trajectory)], axis=0)
        for name, key in (["observations", "state"],
                          ["actions", "action"],
                          ["terminals", "termination"],
                          ["rewards", "cost"],
                          ["next_observations", "next_state"])
    }
    dict_dataset["rewards"] = np.exp(-dict_dataset["rewards"])

    act_low = env.action_space.low.reshape(1, -1)
    act_high = env.action_space.high.reshape(1, -1)
    dict_dataset["actions"] = (dict_dataset["actions"] - act_low) / (act_high - act_low) * 2 - 1
    dict_dataset["observations"] = dict_dataset["observations"][:, :env.observation_space.shape[0]]
    dict_dataset["next_observations"] = dict_dataset["next_observations"][:,
                                                                          :env.observation_space.shape[0]]

    return dict_dataset

