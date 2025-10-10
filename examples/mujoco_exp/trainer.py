from multiprocessing import Value
from typing import Tuple, Optional, List
import os
import argparse
import random
from dataclasses import dataclass, asdict
from collections import namedtuple, defaultdict
from itertools import chain, product
from functools import partial
import datetime
import numpy as np
import torch

from offlinerlkit.utils.logger import Logger
from io_agent.plant.mujoco import Walker2dEnv, HalfCheetahEnv, HopperEnv

from cql import cql_trainer, CqlArgs
from iql import iql_trainer, IqlArgs
from iio import io_trainer, IOArgs
from combo import combo_trainer, ComboArgs
from utils import augment_mujoco_dataset


algorithms = dict(
    cql=dict(trainer=cql_trainer, args=CqlArgs),
    iql=dict(trainer=iql_trainer, args=IqlArgs),
    io=dict(trainer=io_trainer, args=IOArgs),
    combo=dict(trainer=combo_trainer, args=ComboArgs),
)

registered_envs = dict(
        walker2d=Walker2dEnv,
        hopper=HopperEnv,
        cheetah=HalfCheetahEnv
)

def make_augmented_dataset(env_name: str):
    save_dir = f"./data/{env_name}"
    file_name = "augmented.b"
    env_class = registered_envs[env_name]

    if not os.path.exists(os.path.join(save_dir, file_name)):
        env = env_class()
        augment_mujoco_dataset(
            env=env,
            save_dir=save_dir,
            file_name=file_name,
        )


def train_offline_rl(algo_name: str,
                     env_name: str,
                     seed: int,
                     device: str,
                     datasize: int,
                     exp_name: str
                     ) -> None:
    make_augmented_dataset(env_name)
    env = registered_envs[env_name]()
    alg_info = algorithms[algo_name]

    args = alg_info["args"](
        seed=seed,
        obs_shape=env.observation_space.shape,
        action_dim=env.action_space.shape[0],
        device=device,
        datasize=datasize,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    timestamp = datetime.datetime.now().strftime("%y-%m%d-%H%M%S")
    exp_name = "" if exp_name is None else f"{exp_name}/"
    log_dir = f"./logs/{exp_name}{algo_name}/{env_name}/size_{args.datasize}/seed_{seed}/{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dir, output_config)
    logger.log_hyperparameters(vars(args))

    alg_info["trainer"](args, env, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--datasize", type=int, default=int(1e6))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env-name", type=str, default="walker2d")
    parser.add_argument("--algo-name", type=str, default="cql")

    args = parser.parse_args()

    if args.env_name not in registered_envs.keys():
        raise ValueError(f"The given environment: <{args.env_name}>"
            f""" is not found! Please provide one of {", ".join(registered_envs.keys())}.""")

    if args.algo_name not in algorithms.keys():
        raise ValueError(f"The given algorithm: <{args.algo_name}>"
            f""" is not found! Please provide one of {", ".join(algorithms.keys())}.""")

    train_offline_rl(
        datasize=args.datasize,
        device=args.device,
        seed=args.seed,
        env_name=args.env_name,
        algo_name=args.algo_name,
        exp_name=args.exp_name,
    )

