from typing import Tuple, Optional
from dataclasses import dataclass
import numpy as np
import torch
import gym

from offlinerlkit.nets import MLP
from offlinerlkit.modules import Actor, Critic
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.policy_trainer import MFPolicyTrainer
from offlinerlkit.policy import TD3BCPolicy

from io_agent.plant.mujoco import OldSchoolWrapper


@dataclass
class Td3bcArgs():
    seed: int
    obs_shape: Tuple[int, ...]
    action_dim: int
    max_action: float = 1.0
    hidden_dims: Tuple[int, ...] = (256, 256)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    exploration_noise: float = 0.1
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    update_actor_freq: int = 2
    alpha: float = 2.5  # BC weight
    normalize_obs: bool = True
    epoch: int = 100
    step_per_epoch: int = 10000
    eval_episodes: int = 40
    batch_size: int = 32
    datasize: int = int(1e6)
    device: Optional[str] = "cuda" if torch.cuda.is_available() else "cpu"
    data_seed: Optional[int] = None


def td3bc_trainer(args: Td3bcArgs, env: gym.Env, logger: Logger) -> None:

    # create actor
    actor_backbone = MLP(
        input_dim=np.prod(args.obs_shape),
        hidden_dims=args.hidden_dims
    )
    actor = Actor(actor_backbone, args.action_dim, max_action=args.max_action, device=args.device)

    # create critics
    critic1_backbone = MLP(
        input_dim=np.prod(args.obs_shape) + args.action_dim,
        hidden_dims=args.hidden_dims
    )
    critic2_backbone = MLP(
        input_dim=np.prod(args.obs_shape) + args.action_dim,
        hidden_dims=args.hidden_dims
    )
    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)

    # optimizers
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    # load dataset
    dataset = env.env.get_dataset()
    dict_dataset = {}
    keys = ("observations", "actions", "terminals", "rewards", "next_observations")

    if args.data_seed is not None:
        data_rng = np.random.default_rng(args.data_seed)
        n_samples = min(args.datasize, len(dataset["observations"]))
        indices = data_rng.choice(len(dataset["observations"]), size=n_samples, replace=False)
        for key in keys:
            dict_dataset[key] = dataset[key][indices]
    else:
        for key in keys:
            dict_dataset[key] = dataset[key][:args.datasize]

    # normalize observations
    scaler = None
    if args.normalize_obs:
        scaler = StandardScaler()
        scaler.fit(dict_dataset["observations"])

    # create TD3BC policy
    policy = TD3BCPolicy(
        actor,
        critic1,
        critic2,
        actor_optim,
        critic1_optim,
        critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        max_action=args.max_action,
        policy_noise=args.policy_noise * args.max_action,
        noise_clip=args.noise_clip * args.max_action,
        update_actor_freq=args.update_actor_freq,
        alpha=args.alpha,
        scaler=scaler
    )

    print("Number of parameters:", sum(param.numel()
          for key, param in policy.named_parameters()))

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
        eval_episodes=args.eval_episodes
    )
    policy_trainer.train()
