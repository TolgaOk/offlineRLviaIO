from typing import Tuple, Optional, List
import os
import argparse
from dataclasses import dataclass, asdict
from collections import namedtuple, defaultdict
from itertools import chain, product
from functools import partial
import datetime
import numpy as np
import torch
import json
import multiprocessing

from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger
from offlinerlkit.policy_trainer import MFPolicyTrainer
from offlinerlkit.policy import CQLPolicy

from io_agent.plant.base import OldSchoolWrapper
from io_agent.plant.quadrotor import QuadrotorEnv

from common import run_mpc
from utils import parallelize, save_experiment, load_experiment


@dataclass
class Args():
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


def train_quadrotor_cql(n_trajectories: int, device: str, seed: int):

    env = QuadrotorEnv(use_exp_reward=True)
    args = Args(
        seed=seed,
        obs_shape=env.observation_space.shape,
        action_dim=env.action_space.shape[0],
        device=device,
    )

    algo_name: str = "cql"
    task: str = "Quadrotor2d"
    timestamp = datetime.datetime.now().strftime("%y-%m%d-%H%M%S")
    log_dir = f"/mnt/DEPO/tok/sl-to-rl/{task}/{algo_name}/seed_{seed}-size_{n_trajectories}/{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims)
    critic1_backbone = MLP(input_dim=np.prod(args.obs_shape) +
                           args.action_dim, hidden_dims=args.hidden_dims)
    critic2_backbone = MLP(input_dim=np.prod(args.obs_shape) +
                           args.action_dim, hidden_dims=args.hidden_dims)
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True
    )
    actor = ActorProb(actor_backbone, dist, args.device)
    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    policy = CQLPolicy(
        actor,
        critic1,
        critic2,
        actor_optim,
        critic1_optim,
        critic2_optim,
        action_space=env.action_space,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        cql_weight=args.cql_weight,
        temperature=args.temperature,
        max_q_backup=args.max_q_backup,
        deterministic_backup=args.deterministic_backup,
        with_lagrange=args.with_lagrange,
        lagrange_threshold=args.lagrange_threshold,
        cql_alpha_lr=args.cql_alpha_lr,
        num_repeart_actions=args.num_repeat_actions
    )

    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dir, output_config)
    logger.log_hyperparameters(vars(args))

    dataset_trajectories = load_experiment(
        "./quadrotor_data/dataset/mpc_without_hindsight-42")["mpc_trajectories"][:n_trajectories]
    dict_dataset = {
        name: np.stack([asdict(transition)[key]
                       for transition in chain(*dataset_trajectories)], axis=0)
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

    buffer = ReplayBuffer(
        buffer_size=len(dict_dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    buffer.load_dataset(dict_dataset)

    policy_trainer = MFPolicyTrainer(
        policy=policy,
        eval_env=OldSchoolWrapper(env, score_range=(0, 300)),
        buffer=buffer,
        logger=logger,
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        batch_size=args.batch_size,
        eval_episodes=args.eval_episodes
    )
    policy_trainer.train()


def make_policy(args: "Args", env: QuadrotorEnv):
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims)
    critic1_backbone = MLP(input_dim=np.prod(args.obs_shape) +
                           args.action_dim, hidden_dims=args.hidden_dims)
    critic2_backbone = MLP(input_dim=np.prod(args.obs_shape) +
                           args.action_dim, hidden_dims=args.hidden_dims)
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True
    )
    actor = ActorProb(actor_backbone, dist, args.device)
    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    policy = CQLPolicy(
        actor,
        critic1,
        critic2,
        actor_optim,
        critic1_optim,
        critic2_optim,
        action_space=env.action_space,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        cql_weight=args.cql_weight,
        temperature=args.temperature,
        max_q_backup=args.max_q_backup,
        deterministic_backup=args.deterministic_backup,
        with_lagrange=args.with_lagrange,
        lagrange_threshold=args.lagrange_threshold,
        cql_alpha_lr=args.cql_alpha_lr,
        num_repeart_actions=args.num_repeat_actions
    )
    return policy


def evaluate(model: CQLPolicy, env: QuadrotorEnv, trial_seeds: List[int]):
    eps_reward_list = []
    for _seed in trial_seeds:
        state = env.reset(_seed)
        done = False
        eps_reward = 0.0

        while not done:
            action = model.select_action(state.reshape(1, -1), deterministic=True).flatten()
            state, reward, done, info = env.step(action)
            eps_reward += reward
        eps_reward_list.append(eps_reward)
    return eps_reward_list


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trajectories", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    train_quadrotor_cql(
        n_trajectories=args.n_trajectories,
        device=args.device,
        seed=args.seed,
    )
