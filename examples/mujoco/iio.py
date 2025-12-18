## Iterative IO (IIO) agent (Gradient Descent based)

from typing import Tuple, Optional
import numpy as np
from dataclasses import dataclass
import os
import time
from functools import partial
import torch
import jax
import jax.random as jrd
from tqdm import tqdm
import gym

from offlinerlkit.utils.logger import Logger

from io_agent.control.jax_io import JaxIOController
from io_agent.runner.basic import run_agent
from io_agent.utils import load_experiment


@dataclass
class IOArgs:
    seed: int
    obs_shape: Tuple[int, ...]
    action_dim: int
    data_path: str
    num_repeat_actions: int = 10
    learning_rate: float = 5e-2
    lr_exp_decay: float = 0.975
    batch_size: int = 32
    datasize: int = int(1e6)
    data_return_ratio: float = 1.0
    epoch: int = 100
    eval_episodes: int = 40
    step_per_epoch: int = 10000
    device: Optional[str] = "cpu"


def io_trainer(args: IOArgs, env: gym.Env, logger: Logger) -> None:
    # Configure JAX device
    if args.device == "cpu":
        jax.config.update("jax_platforms", "cpu")
    print(f"[DEBUG] JAX devices: {jax.devices()}")

    init_time = time.time()
    rng = np.random.default_rng(args.seed)
    trial_seeds = rng.integers(0, 2**30, args.eval_episodes)

    data = load_experiment(args.data_path)
    augmented_dataset = data["augmented_dataset"]
    feature_handler = data["feature_handler"]

    print("[DEBUG] Creating JaxIOController...")
    t0 = time.time()
    agent = JaxIOController(
        constraints=feature_handler.params.constraints,
        feature_handler=feature_handler,
        key=jrd.PRNGKey(args.seed),
        learning_rate=args.learning_rate,
        include_constraints=True,
        action_constraints_flag=True,
        state_constraints_flag=False,
        lr_exp_decay=args.lr_exp_decay,
        scheduler_transition_step=args.step_per_epoch * (args.epoch // 100),
    )
    print(f"[DEBUG] Create agent: {time.time() - t0:.2f}s")

    best_mean_eval_score = -np.inf
    dataset = augmented_dataset[: int(args.datasize)]
    returns = np.array([transition.info["episode_return"] for transition in dataset])
    indices = np.argsort(returns)[-int(args.data_return_ratio * len(returns)) :]

    print(f"[DEBUG] Creating trainer (jnp.stack of {len(indices)} samples)...")
    t0 = time.time()
    trainer = agent.train(
        [dataset[index] for index in indices], batch_size=args.batch_size
    )
    print(f"[DEBUG] Create trainer: {time.time() - t0:.2f}s")

    print("[DEBUG] First training step (JIT compilation)...")
    t0 = time.time()
    start_time = time.time()
    for epoch in range(args.epoch):
        with tqdm(
            total=args.step_per_epoch, desc=f"Epoch: {epoch + 1}/{args.epoch}"
        ) as pbar:
            steps_done = 0
            while steps_done < args.step_per_epoch:
                step_loss, n_steps = next(trainer)
                if epoch == 0 and steps_done == 0:
                    print(f"[DEBUG] First step (JIT): {time.time() - t0:.2f}s")
                logger.logkv_mean("train/loss", step_loss)
                pbar.set_postfix(
                    {
                        "Step loss": f"{step_loss:.6f}",
                    }
                )
                pbar.update(n_steps)
                steps_done += n_steps

        print(f"[DEBUG] Starting evaluation ({args.eval_episodes} episodes)...")
        t0 = time.time()
        io_trajectories = []
        for i, _seed in enumerate(trial_seeds):
            t_ep = time.time()
            io_trajectories.append(
                partial(
                    run_agent,
                    agent=agent,
                    plant=env,
                    use_foresight=False,
                    bias_aware=False,
                    env_reset_rng=np.random.default_rng(_seed),
                )()
            )
            if i == 0:
                print(f"[DEBUG] First eval episode: {time.time() - t_ep:.2f}s")
        print(f"[DEBUG] Evaluation total: {time.time() - t0:.2f}s")

        io_rewards = [
            np.sum([tran.cost for tran in trajectory]) for trajectory in io_trajectories
        ]
        last_median_eval_score = (
            env.env.get_normalized_score(np.median(io_rewards)) * 100
        )
        last_mean_eval_score = env.env.get_normalized_score(np.mean(io_rewards)) * 100
        last_std_eval_score = env.env.get_normalized_score(np.std(io_rewards)) * 100

        logger.logkv("eval/normalized_episode_reward", last_mean_eval_score)
        logger.logkv("eval/normalized_episode_std", last_std_eval_score)
        logger.logkv("eval/normalized_episode_median", last_median_eval_score)
        logger.logkv("train/lr", agent._last_lr)
        logger.logkv(
            "train/fps",
            ((epoch + 1) * args.step_per_epoch) / (time.time() - start_time),
        )
        logger.set_timestep((epoch + 1) * args.step_per_epoch)
        logger.dumpkvs()

        if last_mean_eval_score > best_mean_eval_score:
            best_mean_eval_score = last_mean_eval_score
            torch.save(
                {
                    "epoch": epoch,
                    "scores": {
                        "median": last_median_eval_score,
                        "mean": last_mean_eval_score,
                        "std": last_std_eval_score,
                    },
                    **agent.state_dict(),
                },
                os.path.join(logger.checkpoint_dir, "policy.pth"),
            )

    logger.log("total time: {:.2f}s".format(time.time() - init_time))
    torch.save(agent.state_dict(), os.path.join(logger.model_dir, "policy.pth"))
    logger.close()
