from functools import partial
import numpy as np
import multiprocessing

from io_agent.plant.quadrotor import QuadrotorEnv
from io_agent.runner.basic import run_mpc
from io_agent.utils import parallelize, save_experiment, load_experiment


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