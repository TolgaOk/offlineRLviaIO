from typing import Tuple, Optional, List
import numpy as np
from dataclasses import dataclass
import os
import time
from functools import partial
import torch
from tqdm import tqdm
import gym
import d4rl

from offlinerlkit.utils.logger import Logger

from io_agent.control.iterative_io import IterativeIOController
from io_agent.runner.basic import run_agent
from io_agent.utils import load_experiment


@dataclass
class IIOArgs():
    seed: int
    obs_shape: Tuple[int, ...]
    action_dim: int
    num_repeat_actions: int = 10
    eval_episodes: int = 20
    learning_rate: float = 5e-2
    lr_exp_decay: float = 0.995
    batch_size: int = 64
    eval_steps: Tuple[int, ...] = tuple(range(0, int(1e6), int(1e3)))
    datasize: int = int(1e6)
    epoch: int = 1000
    step_per_epoch: int = 1000
    data_dir: str = "./data"
    device: Optional[str] = "cuda" if torch.cuda.is_available() else "cpu"


def iio_trainer(args: IIOArgs, env: gym.Env, logger: Logger) -> None:
    init_time = time.time()
    rng = np.random.default_rng(args.seed)
    trial_seeds = rng.integers(0, 2**30, args.eval_episodes)

    task_name = env.__class__.__name__.lower()[:-3]
    walker_data = load_experiment(os.path.join(
        args.data_dir, task_name, "rich_augmented"))
    augmented_dataset = walker_data["augmented_dataset"]
    feature_handler = walker_data["feature_handler"]

    iterative_io_agent = IterativeIOController(
        constraints=feature_handler.params.constraints,
        feature_handler=feature_handler,
        learning_rate=args.learning_rate,
        include_constraints=True,
        action_constraints_flag=True,
        state_constraints_flag=False,
        lr_exp_decay=args.lr_exp_decay,
        device=args.device,
    )

    last_median_eval_score = 0
    trainer = iterative_io_agent.train(
        augmented_dataset[:int(args.datasize)],
        batch_size=args.batch_size,
        rng=rng)
    
    start_time = time.time()
    for epoch in range(args.epoch):
        with tqdm(total=args.step_per_epoch, desc=f"Epoch: {epoch + 1}/{args.epoch}") as pbar:
            for step in range(args.step_per_epoch):
                step_loss = next(trainer)
                logger.logkv_mean("train/loss", step_loss)
                pbar.set_postfix({
                    "Step loss": f"{step_loss:.6f}",
                    "lr": iterative_io_agent.scheduler.get_last_lr()[-1]})
                pbar.update(1)

        iterative_io_trajectories = []
        for _seed in trial_seeds:
            iterative_io_trajectories.append(
                partial(
                    run_agent,
                    agent=iterative_io_agent,
                    plant=env,
                    use_foresight=False,
                    bias_aware=False,
                    env_reset_rng=np.random.default_rng(_seed)
                )()
            )

        iterative_io_rewards = [np.sum([tran.cost for tran in trajectory])
                                for trajectory in iterative_io_trajectories]
        last_median_eval_score = env.env.get_normalized_score(
            np.median(iterative_io_rewards)) * 100
        last_mean_eval_score = env.env.get_normalized_score(
            np.mean(iterative_io_rewards)) * 100
        last_std_eval_score = env.env.get_normalized_score(
            np.std(iterative_io_rewards)) * 100

        logger.logkv(
            "eval/normalized_episode_reward", last_mean_eval_score)
        logger.logkv(
            "eval/normalized_episode_std", last_std_eval_score)
        logger.logkv(
            "eval/normalized_episode_median", last_median_eval_score)
        logger.logkv(
            "train/lr", iterative_io_agent.scheduler.get_last_lr()[-1])
        logger.logkv(
            "train/fps", ((epoch + 1) * args.step_per_epoch) / (time.time() - start_time))
        logger.set_timestep((epoch + 1) * args.step_per_epoch)
        logger.dumpkvs()

        torch.save(iterative_io_agent.state_dict(),
                    os.path.join(logger.checkpoint_dir, "policy.pth"))

    logger.log("total time: {:.2f}s".format(time.time() - init_time))
    torch.save(iterative_io_agent.state_dict(), os.path.join(
        logger.model_dir, "policy.pth"))
    logger.close()
