from typing import Tuple, Optional, List
import numpy as np
import torch
from dataclasses import dataclass
import gym
import d4rl

from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger
from offlinerlkit.policy_trainer import MFPolicyTrainer
from offlinerlkit.policy import CQLPolicy

from io_agent.plant.mujoco import OldSchoolWrapper


@dataclass
class CqlArgs():
    seed: int
    obs_shape: Tuple[int, ...]
    action_dim: int
    hidden_dims: Tuple[int, ...] = (256, 256, 256)
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
    epoch: int = 1000
    step_per_epoch: int = 1000
    eval_episodes: int = 40
    batch_size: int = 256
    datasize: int = int(1e6)
    device: Optional[str] = "cuda" if torch.cuda.is_available() else "cpu"


def cql_trainer(args: CqlArgs, env: gym.Env, logger: Logger) -> MFPolicyTrainer:

    actor_backbone = MLP(input_dim=np.prod(
        args.obs_shape), hidden_dims=args.hidden_dims)
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
    critic1_optim = torch.optim.Adam(
        critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(
        critic2.parameters(), lr=args.critic_lr)

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
    print("Number of parameters:", sum(param.numel()
          for key, param in policy.named_parameters()))

    dataset = env.env.get_dataset()
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

    policy_trainer = MFPolicyTrainer(
        policy=policy,
        eval_env=OldSchoolWrapper(env),
        buffer=buffer,
        logger=logger,
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        batch_size=args.batch_size,
        eval_episodes=args.eval_episodes
    )
    return policy_trainer
