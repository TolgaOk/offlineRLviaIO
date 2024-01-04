from typing import Tuple, Optional, List
import numpy as np
import torch
from dataclasses import dataclass
import gym
import d4rl

from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian, EnsembleDynamicsModel
from offlinerlkit.dynamics import EnsembleDynamics
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.termination_fns import get_termination_fn
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger
from offlinerlkit.policy_trainer import MBPolicyTrainer
from offlinerlkit.policy import COMBOPolicy

from io_agent.plant.mujoco import OldSchoolWrapper


@dataclass
class ComboArgs():
    seed: int
    obs_shape: Tuple[int, ...]
    action_dim: int
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    hidden_dims: Tuple[int, ...] = (256, 256, 256)
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    auto_alpha: bool = True
    target_entropy: int = None
    alpha_lr: float = 1e-4
    cql_weight: float = 5.0
    temperature: float = 1.0
    max_q_backup: bool = False
    deterministic_backup: bool = True
    with_lagrange: bool = False
    lagrange_threshold: float = 10.0
    cql_alpha_lr: float = 3e-4
    num_repeat_actions: int = 10
    uniform_rollout: bool = False
    rho_s: str = "mix"
    dynamics_lr: float = 1e-3
    dynamics_hidden_dims: Tuple[int, ...] = (200, 200, 200, 200)
    dynamics_weight_decay: Tuple[float, ...] = (2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4)
    n_ensemble: int = 7
    n_elites: int = 5
    rollout_freq: int = 1000
    rollout_batch_size: int = 50000
    rollout_length: int = 5
    model_retain_epochs: int = 5
    real_ratio: float = 0.5
    load_dynamics_path: str = None
    epoch: int = 100
    step_per_epoch: int = 10000
    eval_episodes: int = 40
    batch_size: int = 256
    datasize: int = int(1e6)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def combo_trainer(args: ComboArgs, env: gym.Env, logger: Logger) -> None:
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims)
    critic1_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    critic2_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
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

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim, args.epoch)

    if args.auto_alpha:
        target_entropy = args.target_entropy if args.target_entropy \
            else -np.prod(env.action_space.shape)
        args.target_entropy = target_entropy
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args.alpha

    # create dynamics
    load_dynamics_model = True if args.load_dynamics_path else False
    dynamics_model = EnsembleDynamicsModel(
        obs_dim=np.prod(args.obs_shape),
        action_dim=args.action_dim,
        hidden_dims=args.dynamics_hidden_dims,
        num_ensemble=args.n_ensemble,
        num_elites=args.n_elites,
        weight_decays=args.dynamics_weight_decay,
        device=args.device
    )
    dynamics_optim = torch.optim.Adam(
        dynamics_model.parameters(),
        lr=args.dynamics_lr
    )
    scaler = StandardScaler()
    termination_fn = get_termination_fn(task=env.task_name)
    dynamics = EnsembleDynamics(
        dynamics_model,
        dynamics_optim,
        scaler,
        termination_fn
    )

    if args.load_dynamics_path:
        dynamics.load(args.load_dynamics_path)

    # create policy
    policy = COMBOPolicy(
        dynamics,
        actor,
        critic1,
        critic2,
        actor_optim,
        critic1_optim,
        critic2_optim,
        action_space=env.action_space,
        tau=args.tau,
        gamma=args.gamma,
        alpha=alpha,
        cql_weight=args.cql_weight,
        temperature=args.temperature,
        max_q_backup=args.max_q_backup,
        deterministic_backup=args.deterministic_backup,
        with_lagrange=args.with_lagrange,
        lagrange_threshold=args.lagrange_threshold,
        cql_alpha_lr=args.cql_alpha_lr,
        num_repeart_actions=args.num_repeat_actions,
        uniform_rollout=args.uniform_rollout,
        rho_s=args.rho_s
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

    
    # create buffer
    real_buffer = ReplayBuffer(
        buffer_size=len(dict_dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    real_buffer.load_dataset(dict_dataset)
    fake_buffer = ReplayBuffer(
        buffer_size=args.rollout_batch_size*args.rollout_length*args.model_retain_epochs,
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )

    # create policy trainer
    policy_trainer = MBPolicyTrainer(
        policy=policy,
        eval_env=OldSchoolWrapper(env),
        real_buffer=real_buffer,
        fake_buffer=fake_buffer,
        logger=logger,
        rollout_setting=(args.rollout_freq, args.rollout_batch_size, args.rollout_length),
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        batch_size=args.batch_size,
        real_ratio=args.real_ratio,
        eval_episodes=args.eval_episodes,
        lr_scheduler=lr_scheduler
    )

    # train
    if not load_dynamics_model:
        dynamics.train(real_buffer.sample_all(), logger, max_epochs_since_update=5)
    
    policy_trainer.train()
