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
from combo import combo_trainer, ComboArgs
from mopo import mopo_trainer, MopoArgs
from td3bc import td3bc_trainer, Td3bcArgs


algorithms = dict(
    cql=dict(trainer=cql_trainer, args=CqlArgs),
    iql=dict(trainer=iql_trainer, args=IqlArgs),
    combo=dict(trainer=combo_trainer, args=ComboArgs),
    mopo=dict(trainer=mopo_trainer, args=MopoArgs),
    td3bc=dict(trainer=td3bc_trainer, args=Td3bcArgs),
)

registered_envs = dict(
        walker2d=Walker2dEnv,
        hopper=HopperEnv,
        cheetah=HalfCheetahEnv
)

def train_offline_rl(algo_name: str,
                     env_name: str,
                     seed: int,
                     device: str,
                     datasize: int,
                     exp_name: str,
                     batch_size: int = 32,
                     data_seed: int = None,
                     ) -> None:
    env = registered_envs[env_name]()
    alg_info = algorithms[algo_name]

    # Base args for all algorithms
    base_args = dict(
        seed=seed,
        obs_shape=env.observation_space.shape,
        action_dim=env.action_space.shape[0],
        device=device,
        datasize=datasize,
        batch_size=batch_size,
        data_seed=data_seed,
    )

    # Add max_action for MOPO and TD3BC
    if algo_name in ["mopo", "td3bc"]:
        base_args["max_action"] = float(env.action_space.high[0])

    args = alg_info["args"](**base_args)

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
    parser.add_argument("--env-name", type=str, default="hopper")
    parser.add_argument("--algo-name", type=str, default="cql")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-seed", type=int, default=None,
                        help="Seed for random data sampling (default: use first N samples)")

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
        batch_size=args.batch_size,
        data_seed=args.data_seed,
    )

