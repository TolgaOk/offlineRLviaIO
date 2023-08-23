from typing import List, Tuple
from functools import partial
import datetime
import os
from dataclasses import dataclass, asdict
import json
from tqdm.notebook import tqdm
import numpy as np
import torch

from io_agent.plant.base import Plant
from io_agent.plant.mujoco import MuJoCoEnv
from io_agent.evaluator import Transition
from io_agent.control.io import FeatureHandler, AugmentDataset
from io_agent.control.iterative_io import IterativeIOController
from io_agent.runner.basic import run_agent
from io_agent.utils import save_experiment, load_experiment


@dataclass
class IterativeIOArgs():
    work_dir: str
    learning_rate: float = 1e-2
    lr_exp_decay: float = 0.98
    n_batch: int = 128
    data_size: int = int(1e6)
    eval_epochs: Tuple[int] = tuple(range(20, 2))


def run_iterative_io(args: IterativeIOArgs,
                     env: Plant,
                     seed: int,
                     trial_seeds: List[int],
                     name: str,
                     save_dir: str = "models",
                     log_dir: str = "logs",
                     data_path: str = "dataset/rich_augmented",
                     device: str = "cpu",
                     verbose: bool = True,
                     ):

    timestamp = datetime.datetime.now().strftime("%y-%m%d-%H%M%S")
    save_dir = os.path.join(args.work_dir, save_dir, f"{timestamp}")
    log_dir = os.path.join(args.work_dir, log_dir, f"{timestamp}")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    walker_data = load_experiment(os.path.join(args.work_dir, data_path))
    augmented_dataset = walker_data["augmented_dataset"]
    feature_handler = walker_data["feature_handler"]

    rng = np.random.default_rng(seed)
    model_seed = rng.integers(0, 2**30).item()
    torch.manual_seed(model_seed)
    iterative_io_agent = IterativeIOController(
        constraints=feature_handler.params.constraints,
        feature_handler=feature_handler,
        learning_rate=args.learning_rate,
        include_constraints=True,
        action_constraints_flag=True,
        state_constraints_flag=False,
        lr_exp_decay=args.lr_exp_decay,
        device=device,
    )
    epoch_losses = []
    step_losses = []
    costs = {}
    last_median_eval_score = None
    trainer = iterative_io_agent.train(augmented_dataset[:int(args.data_size)],
                                       batch_size=args.n_batch,
                                       rng=rng)
    with open(os.path.join(log_dir, f"{name}_args.json"), "w") as fobj:
        json.dump({
            "args": asdict(args),
            "name": name,
            "seeds": {
                "trials": [value.item() for value in trial_seeds],
                "model": model_seed,
                "train": seed.item()}
        }, fobj)
    with tqdm(total=args.eval_epochs[-1]) as pbar:
        for eval_break_epoch in args.eval_epochs:
            avg_epoch_loss = None
            for epoch in range(len(epoch_losses), eval_break_epoch):
                avg_epoch_loss, _step_losses = next(trainer)
                epoch_losses.append(avg_epoch_loss)
                step_losses.extend(_step_losses)

                pbar.set_postfix({
                    "Median score": f"{last_median_eval_score:.3f}%",
                    "Epoch loss": f"{avg_epoch_loss:.6f}",
                    "LR": iterative_io_agent.scheduler.get_last_lr()[-1]})
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
            costs[eval_break_epoch] = iterative_io_rewards
            last_median_eval_score = env.env.get_normalized_score(
                np.median(iterative_io_rewards)) * 100
            last_mean_eval_score = env.env.get_normalized_score(np.mean(iterative_io_rewards)) * 100
            last_std_eval_score = env.env.get_normalized_score(np.std(iterative_io_rewards)) * 100
            torch.save(iterative_io_agent, os.path.join(
                save_dir, f"model_{name}_{eval_break_epoch}_{model_seed}"))
            if verbose:
                print(f"median score: {last_median_eval_score:.3f}%",
                      f"average score: {last_mean_eval_score:.3f}%",
                      f"std score: {last_std_eval_score:.3f}%")
            with open(os.path.join(log_dir, name + f"_logs_{eval_break_epoch}.json"), "a") as fobj:
                json.dump({"rewards": iterative_io_rewards,
                           "Median score": f"{last_median_eval_score:.3f}%",
                           "Average score": f"{last_mean_eval_score:.3f}%",
                           "Std score": f"{last_std_eval_score:.3f}%",
                           "Epoch loss": avg_epoch_loss,
                           "LR": iterative_io_agent.scheduler.get_last_lr()[-1]
                           }, fobj)
    return costs, epoch_losses, step_losses, iterative_io_agent


def augment_mujoco_dataset(
        env: MuJoCoEnv,
        save_dir: str,
        file_name: str,
        n_past: int = 4,
        use_co_product: bool = True,
        add_bias: bool = True,
        use_sinusoidal: bool = True,
        use_action_regressor: bool = True,
        use_state_regressor: bool = True) -> None:
    dataset = env.env.get_dataset()
    feature_handler = FeatureHandler(
        params=env.nominal_model(),
        n_past=n_past,
        use_co_product=use_co_product,
        add_bias=add_bias,
        scale_factor=1.0,
        use_sinusoidal=use_sinusoidal,
        use_action_regressor=use_action_regressor,
        use_noise_regressor=False,
        use_state_regressor=use_state_regressor,
        state_high=dataset["observations"].max(axis=0),
        state_low=dataset["observations"].min(axis=0),
        noise_size=0,
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.shape[0],
        output_size=env.observation_space.shape[0],
    )
    augmenter = AugmentDataset(
        expert_agent=None,
        feature_handler=feature_handler
    )

    trajectories = []

    states = dataset["observations"]
    actions = dataset["actions"]

    dones = np.argwhere(np.logical_or(dataset["terminals"], dataset["timeouts"]))

    start_index = 0
    for done_index in dones:
        done_index = done_index.item()
        trajectory = []
        for index in range(start_index, done_index + 1):
            trajectory.append(
                Transition(
                    state=states[index],
                    action=actions[index],
                    next_state=None,
                    cost=None,
                    termination=None,
                    truncation=None,
                    info=None,
                    reference=None,
                )
            )
        start_index = done_index + 1
        trajectories.append(trajectory)

    # Sanity check
    start_indices = np.concatenate(
        [np.zeros((1,), dtype=np.int32), dones.ravel()[:-1].astype(np.int32) + 1])
    traj_lengths = (dones + 1).ravel() - start_indices
    assert np.all([len(traj) == traj_lengths[index] for index, traj in enumerate(trajectories)])

    augmented_trajectories = augmenter(trajectories)
    save_experiment(
        values={
            "augmented_dataset": augmented_trajectories,
            "feature_handler": feature_handler},
        seed=None,  # The process is deterministic
        exp_dir=save_dir,
        name=file_name
    )
