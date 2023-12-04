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

from cql import cql_trainer, CqlArgs
from iql import iql_trainer, IqlArgs
from iio import iio_trainer, IIOArgs

from io_agent.plant.mujoco import Walker2dEnv, HalfCheetahEnv, HopperEnv


algorithms = dict(
    cql=dict(trainer=cql_trainer, args=CqlArgs),
    iql=dict(trainer=iql_trainer, args=IqlArgs),
    io=dict(trainer=iio_trainer, args=IIOArgs),
)


def train_offline_rl(algo_name: str,
                     env_name: str,
                     seed: int,
                     device: str,
                     datasize: int
                     ) -> None:

    if env_name == "walker":
        env = Walker2dEnv()
    elif env_name == "cheetah":
        env = HalfCheetahEnv()
    elif env_name == "hopper":
        env = HopperEnv()

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
    # env.seed(args.seed)

    timestamp = datetime.datetime.now().strftime("%y-%m%d-%H%M%S")
    log_dir = f"./logs/{algo_name}/{env_name}/size_{args.datasize}/seed_{seed}/{timestamp}"
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
    parser.add_argument("--datasize", type=int, default=int(1e6))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env-name", type=str, default="walker")
    parser.add_argument("--algo-name", type=str, default="cql")

    args = parser.parse_args()

    train_offline_rl(
        datasize=args.datasize,
        device=args.device,
        seed=args.seed,
        env_name=args.env_name,
        algo_name=args.algo_name,
    )
