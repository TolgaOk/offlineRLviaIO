from typing import Tuple
from dataclasses import dataclass
import numpy as np
import torch
import gym
import d4rl

from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, Critic, DiagGaussian
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger
from offlinerlkit.policy_trainer import MFPolicyTrainer
from offlinerlkit.policy import IQLPolicy

from io_agent.plant.mujoco import OldSchoolWrapper


@dataclass
class IqlArgs():
    seed: int
    obs_shape: Tuple[int, ...]
    action_dim: int
    hidden_dims: Tuple[int, ...] = (256, 256)
    actor_lr: float = 3e-4
    critic_q_lr: float = 3e-4
    critic_v_lr: float = 3e-4
    dropout_rate: float = None
    lr_decay: bool = True
    gamma: float = 0.99
    tau: float = 0.005
    expectile: float = 0.7
    temperature: float = 3.0
    epoch: int = 1000
    step_per_epoch: int = 1000
    eval_episodes: int = 10
    batch_size: int = 256
    datasize: int = int(1e6)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def normalize_rewards(dataset):
    terminals_float = np.zeros_like(dataset["rewards"])
    for i in range(len(terminals_float) - 1):
        if np.linalg.norm(dataset["observations"][i + 1] -
                          dataset["next_observations"][i]
                          ) > 1e-6 or dataset["terminals"][i] == 1.0:
            terminals_float[i] = 1
        else:
            terminals_float[i] = 0

    terminals_float[-1] = 1

    # split_into_trajectories
    trajs = [[]]
    for i in range(len(dataset["observations"])):
        trajs[-1].append((dataset["observations"][i], dataset["actions"][i], dataset["rewards"][i], 1.0-dataset["terminals"][i],
                          terminals_float[i], dataset["next_observations"][i]))
        if terminals_float[i] == 1.0 and i + 1 < len(dataset["observations"]):
            trajs.append([])

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)

    # normalize rewards
    dataset["rewards"] /= compute_returns(trajs[-1]) - \
        compute_returns(trajs[0])
    dataset["rewards"] *= 1000.0

    return dataset


def iql_trainer(args: IqlArgs, env: gym.Env, logger: Logger) -> None:

    # create policy model
    actor_backbone = MLP(input_dim=np.prod(
        args.obs_shape), hidden_dims=args.hidden_dims, dropout_rate=args.dropout_rate)
    critic_q1_backbone = MLP(input_dim=np.prod(
        args.obs_shape)+args.action_dim, hidden_dims=args.hidden_dims)
    critic_q2_backbone = MLP(input_dim=np.prod(
        args.obs_shape)+args.action_dim, hidden_dims=args.hidden_dims)
    critic_v_backbone = MLP(input_dim=np.prod(
        args.obs_shape), hidden_dims=args.hidden_dims)
    dist = DiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=False,
        conditioned_sigma=False
    )
    actor = ActorProb(actor_backbone, dist, args.device)
    critic_q1 = Critic(critic_q1_backbone, args.device)
    critic_q2 = Critic(critic_q2_backbone, args.device)
    critic_v = Critic(critic_v_backbone, args.device)

    for m in list(actor.modules()) + list(critic_q1.modules()) + list(critic_q2.modules()) + list(critic_v.modules()):
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)

    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_q1_optim = torch.optim.Adam(
        critic_q1.parameters(), lr=args.critic_q_lr)
    critic_q2_optim = torch.optim.Adam(
        critic_q2.parameters(), lr=args.critic_q_lr)
    critic_v_optim = torch.optim.Adam(
        critic_v.parameters(), lr=args.critic_v_lr)

    if args.lr_decay:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            actor_optim, args.epoch)
    else:
        lr_scheduler = None

    # create IQL policy
    policy = IQLPolicy(
        actor,
        critic_q1,
        critic_q2,
        critic_v,
        actor_optim,
        critic_q1_optim,
        critic_q2_optim,
        critic_v_optim,
        action_space=env.action_space,
        tau=args.tau,
        gamma=args.gamma,
        expectile=args.expectile,
        temperature=args.temperature
    )

    print("Number of parameters:", sum(param.numel()
          for key, param in policy.named_parameters()))

    dataset = env.env.get_dataset()
    dataset = normalize_rewards(dataset)
    dict_dataset = {}

    for key in ("observations",
                "actions",
                "terminals",
                "rewards",
                "next_observations"):
        dict_dataset[key] = dataset[key][:args.datasize]

    buffer = ReplayBuffer(
        buffer_size=len(dict_dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    buffer.load_dataset(dict_dataset)

    # create policy trainer
    policy_trainer = MFPolicyTrainer(
        policy=policy,
        eval_env=OldSchoolWrapper(env),
        buffer=buffer,
        logger=logger,
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        batch_size=args.batch_size,
        eval_episodes=args.eval_episodes,
        lr_scheduler=lr_scheduler
    )
    policy_trainer.train()
