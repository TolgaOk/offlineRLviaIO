from typing import Tuple, Optional, List
import numpy as np
from dataclasses import dataclass
import os
import time
from functools import partial
import torch
import jax.random as jrd
from tqdm import tqdm
import gym
import d4rl

from offlinerlkit.utils.logger import Logger

from io_agent.control.jax_io import JaxIOController
from io_agent.runner.basic import run_agent
from io_agent.utils import load_experiment


@dataclass
class IIOArgs():
    seed: int
    obs_shape: Tuple[int, ...]
    action_dim: int
    num_repeat_actions: int = 10
    learning_rate: float = 5e-2
    lr_exp_decay: float = 0.975
    batch_size: int = 32
    datasize: int = int(1e6)
    data_return_ratio: float = 1.0
    epoch: int = 100
    eval_episodes: int = 40
    step_per_epoch: int = 10000
    data_dir: str = "./data"
    device: Optional[str] = "auto"


def iio_trainer(args: IIOArgs, env: gym.Env, logger: Logger) -> None:
    init_time = time.time()
    rng = np.random.default_rng(args.seed)
    trial_seeds = rng.integers(0, 2**30, args.eval_episodes)

    task_name = env.__class__.__name__.lower()[:-3]
    walker_data = load_experiment(os.path.join(
        args.data_dir, task_name, "rich_augmented_v3"))
    augmented_dataset = walker_data["augmented_dataset"]
    feature_handler = walker_data["feature_handler"]

    iterative_io_agent = JaxIOController(
        constraints=feature_handler.params.constraints,
        feature_handler=feature_handler,
        key=jrd.PRNGKey(args.seed),
        learning_rate=args.learning_rate,
        include_constraints=True,
        action_constraints_flag=True,
        state_constraints_flag=False,
        lr_exp_decay=args.lr_exp_decay,
        scheduler_transition_step=args.step_per_epoch * (args.epoch // 100)
    )

    best_mean_eval_score = -np.inf
    dataset = augmented_dataset[:int(args.datasize)]
    returns = np.array([transition.info["episode_return"] for transition in dataset])
    indices = np.argsort(returns)[-int(args.data_return_ratio * len(returns)):]

    trainer = iterative_io_agent.train(
        [dataset[index] for index in indices],
        batch_size=args.batch_size)

    start_time = time.time()
    for epoch in range(args.epoch):
        with tqdm(total=args.step_per_epoch, desc=f"Epoch: {epoch + 1}/{args.epoch}") as pbar:
            for step in range(args.step_per_epoch):
                step_loss = next(trainer)
                logger.logkv_mean("train/loss", step_loss)
                pbar.set_postfix({
                    "Step loss": f"{step_loss:.6f}",
                })
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
            "train/lr", iterative_io_agent._last_lr)
        logger.logkv(
            "train/fps", ((epoch + 1) * args.step_per_epoch) / (time.time() - start_time))
        logger.set_timestep((epoch + 1) * args.step_per_epoch)
        logger.dumpkvs()

        if last_mean_eval_score > best_mean_eval_score:
            best_mean_eval_score = last_mean_eval_score
            torch.save({"epoch": epoch,
                        "scores": {
                            "median": last_median_eval_score,
                            "mean": last_mean_eval_score,
                            "std": last_std_eval_score,
                        },
                        **iterative_io_agent.state_dict()},
                       os.path.join(logger.checkpoint_dir, "policy.pth"))

    logger.log("total time: {:.2f}s".format(time.time() - init_time))
    torch.save(iterative_io_agent.state_dict(), os.path.join(
        logger.model_dir, "policy.pth"))
    logger.close()
