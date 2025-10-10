from functools import partial
import numpy as np

from io_agent.plant.base import OldSchoolWrapper
import torch
from collections import defaultdict
from itertools import product, chain
import os
import json
from typing import Tuple, NamedTuple, List
from io_agent.plant.quadrotor import QuadrotorEnv
from io_agent.utils import parallelize, save_experiment, load_experiment
from io_agent.plant.base import OldSchoolWrapper
from itertools import product
import random
import datetime
import os

from mujoco_exp.cql import cql_trainer, CqlArgs
from io_agent.plant.quadrotor import QuadrotorEnv
from offlinerlkit.utils.logger import Logger
from utils import trajectory_to_dataset
from io_agent.utils import load_experiment



@dataclass
class CqlArgs():
    seed: int
    obs_shape: Tuple[int]
    action_dim: int
    hidden_dims: Tuple[int] = (256, 256, 256)
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    target_entropy: int = -6.0
    auto_alpha = True
    alpha_lr: float = 1e-4
    cql_weight: float = 5.0
    temperature: float = 1.0
    max_q_backup: bool = False
    deterministic_backup: bool = True
    with_lagrange: bool = False
    lagrange_threshold: float = 10.0
    cql_alpha_lr: float = 3e-4
    num_repeat_actions: int = 10
    epoch: int = 50
    step_per_epoch: int = 1000
    eval_episodes: int = 10
    batch_size: int = 256
    device: Optional[str] = "cuda:0"


def cql_quadrotor_trainer(args: CqlArgs, env: QuadrotorEnv, logger: Logger, dataset: Optional[dict] = None) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = QuadrotorEnv(use_exp_reward=True)

    for seed, n_trajectories in product([42, 43, 44, 45], [100, 80, 50, 25, 20, 15, 12, 10]):
        args = CqlArgs(
            seed=seed,
            obs_shape=env.observation_space.shape,
            action_dim=env.action_space.shape[0],
            device=device,
        )    

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

        timestamp = datetime.datetime.now().strftime("%y-%m%d-%H%M%S")
        algo_name: str = "cql"
        task: str = "Quadrotor2d"
        log_dir = f"./logs/offlineRL_data/{task}/{algo_name}/seed_{seed}-size_{n_trajectories}/{timestamp}"
        os.makedirs(log_dir, exist_ok=True)

        output_config = {
            "consoleout_backup": "stdout",
            "policy_training_progress": "csv",
            "tb": "tensorboard"
        }
        logger = Logger(log_dir, output_config)
        logger.log_hyperparameters(vars(args))
        trajectories = load_experiment(
            "./quadrotor_data/dataset/mpc_without_hindsight-42")["mpc_trajectories"][:n_trajectories]
        dataset = trajectory_to_dataset(trajectories, env)
        cql_trainer(
            args=args,
            env=env,
            logger=logger,
            dataset=dataset
        )




def evaluate_all_cql_quadrotor_policies(experiment_path: str,
                                        cql_seeds: Tuple[int],
                                        data_sizes: Tuple[int],
                                        epochs_numbers: Tuple[int],
                                        initial_seed: int,
                                        n_trials: int,
                                        n_proc: int
                                        ) -> None:

    def _evaluate_epochs(cql_seed: int, data_size: int) -> None:
        path = os.path.join(experiment_path, f"seed_{cql_seed}-size_{data_size}/")
        latest_experiment_name = sorted(os.listdir(path))[-1]

        checkpoint_dir = os.path.join(path, latest_experiment_name, "checkpoint")
        argument_path = os.path.join(path, latest_experiment_name, "record", "hyper_param.json")
        with open(argument_path, "r") as fobj:
            arg_dict = json.load(fobj)
            Args = namedtuple("Args", ",".join(arg_dict.keys()))
            args = Args(**arg_dict)
        args

        seed_rng = np.random.default_rng(initial_seed)
        trial_seeds = [seed.item() for seed in seed_rng.integers(0, 2**30, n_trials)]
        env = QuadrotorEnv(use_exp_reward=True)
        env = OldSchoolWrapper(env, score_range=(0, 300))

        model_evaluations = {}
        model = make_policy(args, env)
        for index in epochs_numbers:
            model.load_state_dict(torch.load(
                os.path.join(checkpoint_dir, "policy_{0:04d}.pth".format(index))))
            model_evaluations[index] = evaluate(model, env, trial_seeds)
        return model_evaluations

    key_pairs = list(product(cql_seeds, data_sizes))
    evaluations = parallelize(
        n_proc=n_proc,
        fn=_evaluate_epochs,
        kwargs_list=[
            dict(cql_seed=cql_seed, data_size=data_size)
            for cql_seed, data_size in key_pairs
        ],
        loading_bar_kwargs=dict(desc="Progress")
    )

    cql_evaluations = defaultdict(dict)
    for eval_scores, (cql_seed, data_size) in zip(evaluations, key_pairs):
        cql_evaluations[data_size][cql_seed] = eval_scores

    return cql_evaluations