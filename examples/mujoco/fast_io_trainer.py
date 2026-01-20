"""Fast IO trainer with on-the-fly augmentation matching FeatureHandler."""
/039/󰌠 /fast_io_trainer.py

from typing import Tuple, Optional
import numpy as np
from dataclasses import dataclass
import os
import time

# Disable JAX GPU memory preallocation (use memory as needed)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import torch
import jax
import jax.numpy as jnp
import jax.random as jrd
import optax
import gym

from offlinerlkit.utils.logger import Logger
from io_agent.control.fast_io import Augmenter, io_init, train_epoch
from io_agent.control.fast_eval import evaluate, _make_eval_fns, run_episode


def filter_top_trajectories(dataset: dict, top_pct: float = 0.1) -> np.ndarray:
    """Filter dataset to top trajectories by return.

    Args:
        dataset: D4RL dataset dict with observations, actions, rewards, terminals, timeouts
        top_pct: Fraction of trajectories to keep (e.g., 0.1 for top 10%)

    Returns:
        indices: Array of indices for samples in top trajectories
    """
    terminals = dataset["terminals"]
    timeouts = dataset.get("timeouts", np.zeros_like(terminals))
    rewards = dataset["rewards"]

    # Find trajectory boundaries
    dones = np.logical_or(terminals, timeouts)
    traj_ends = np.where(dones)[0]
    traj_starts = np.concatenate([[0], traj_ends[:-1] + 1])

    # Calculate return per trajectory
    traj_returns = np.array([
        rewards[start:end+1].sum()
        for start, end in zip(traj_starts, traj_ends)
    ])

    # Find threshold for top percentile
    threshold = np.percentile(traj_returns, (1 - top_pct) * 100)
    top_mask = traj_returns >= threshold

    # Collect indices from top trajectories
    indices = []
    for i, (start, end) in enumerate(zip(traj_starts, traj_ends)):
        if top_mask[i]:
            indices.extend(range(start, end + 1))

    print(f"Top {top_pct*100:.0f}% filter: {top_mask.sum()}/{len(traj_returns)} trajs, {len(indices)} samples")
    print(f"Return threshold: {threshold:.1f}, range: [{traj_returns[top_mask].min():.1f}, {traj_returns.max():.1f}]")

    return np.array(indices)


@dataclass
class FastIOArgs:
    seed: int
    obs_shape: Tuple[int, ...]
    action_dim: int
    learning_rate: float = 5e-2
    lr_exp_decay: float = 0.975
    seq_len: int = 64
    batch_size: int = 32
    step_per_epoch: int = 10000
    datasize: int = int(1e6)
    epoch: int = 100
    eval_episodes: int = 40
    n_past: int = 4
    scale_factor: float = 1.0
    add_bias: bool = True
    use_state_regressor: bool = True
    use_action_regressor: bool = True
    use_co_product: bool = True
    use_sinusoidal: bool = True
    sin_freqs: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
    use_rff: bool = True
    rff_n_features: int = 1024
    rff_sigma: float = 0.1
    rff_seed: int = 42
    use_rbf: bool = False
    rbf_n_features: int = 1024
    rbf_sigma: float = 0.1
    norm_momentum: float = 0.0
    use_adam: bool = False
    top_return_pct: float = 1.0
    device: Optional[str] = "cpu"
    data_seed: Optional[int] = None


def fast_io_trainer(args: FastIOArgs, env: gym.Env, logger: Logger) -> None:
    if args.device == "cpu":
        jax.config.update("jax_platforms", "cpu")

    init_time = time.time()
    rng = np.random.default_rng(args.seed)
    trial_seeds = rng.integers(0, 2**30, args.eval_episodes)

    dataset = env.env.get_dataset()
    obs_all = dataset["observations"]
    actions_all = dataset["actions"]
    terminals = dataset["terminals"]
    timeouts = dataset.get("timeouts", np.zeros_like(terminals))

    # Filter to top trajectories if requested
    if args.top_return_pct < 1.0:
        available_indices = filter_top_trajectories(dataset, args.top_return_pct)
    else:
        available_indices = np.arange(len(obs_all))

    n_samples = min(args.datasize, len(available_indices))
    if args.data_seed is not None:
        data_rng = np.random.default_rng(args.data_seed)
        indices = data_rng.choice(available_indices, size=n_samples, replace=False)
        indices = np.sort(indices)
    else:
        indices = available_indices[:n_samples]

    obs = jnp.asarray(obs_all[indices])
    actions = jnp.asarray(actions_all[indices])

    dones = jnp.asarray(
        np.logical_or(terminals[indices], timeouts[indices]).astype(np.float32)
    )

    print(f"Data: {obs.shape[0]} samples, {int(dones.sum())} episodes")

    state_low, state_high = obs_all.min(axis=0), obs_all.max(axis=0)
    augmenter = Augmenter.create(
        obs_dim=args.obs_shape[0],
        act_dim=args.action_dim,
        state_low=state_low,
        state_high=state_high,
        n_past=args.n_past,
        scale_factor=args.scale_factor,
        add_bias=args.add_bias,
        use_state_regressor=args.use_state_regressor,
        use_action_regressor=args.use_action_regressor,
        use_co_product=args.use_co_product,
        use_sinusoidal=args.use_sinusoidal,
        sin_freqs=args.sin_freqs,
        use_rff=args.use_rff,
        rff_n_features=args.rff_n_features,
        rff_sigma=args.rff_sigma,
        rff_seed=args.rff_seed,
        use_rbf=args.use_rbf,
        rbf_n_features=args.rbf_n_features,
        rbf_sigma=args.rbf_sigma,
        norm_momentum=args.norm_momentum,
    )

    aug_dim = augmenter.aug_dim()
    print(f"Augmented dim: {aug_dim}")

    schedule = optax.exponential_decay(
        init_value=args.learning_rate,
        transition_steps=args.step_per_epoch,
        decay_rate=args.lr_exp_decay,
    )
    if args.use_adam:
        optimizer = optax.adam(schedule)
    else:
        optimizer = optax.sgd(schedule)

    io_state = io_init(aug_dim, args.action_dim, jrd.PRNGKey(args.seed), optimizer)

    @jax.jit
    def train_step(io_state):
        return train_epoch(io_state, obs, actions, dones, augmenter, optimizer,
                           args.seq_len, args.batch_size, args.step_per_epoch)

    # Pre-compile eval functions
    augment_obs, update_history, get_action = _make_eval_fns(augmenter)

    best_score = -np.inf
    start_time = time.time()

    for epoch in range(args.epoch):
        t0 = time.time()
        io_state, loss = train_step(io_state)
        loss.block_until_ready()
        train_time = time.time() - t0

        t0 = time.time()
        rewards = np.array([
            run_episode(io_state.theta_su, env, int(s), augmenter, io_state.norm_state, augment_obs, update_history, get_action)
            for s in trial_seeds
        ])
        eval_time = time.time() - t0

        mean_r = env.env.get_normalized_score(np.mean(rewards)) * 100
        std_r = env.env.get_normalized_score(np.std(rewards)) * 100
        median_r = env.env.get_normalized_score(np.median(rewards)) * 100

        logger.logkv("train/loss", float(loss))
        logger.logkv("train/train_time", train_time)
        logger.logkv("train/eval_time", eval_time)
        logger.logkv("train/fps", ((epoch + 1) * args.step_per_epoch) / (time.time() - start_time))
        logger.logkv("eval/normalized_episode_reward", mean_r)
        logger.logkv("eval/normalized_episode_std", std_r)
        logger.logkv("eval/normalized_episode_median", median_r)
        logger.logkv("train/wall_time", time.time() - start_time)
        logger.set_timestep((epoch + 1) * args.step_per_epoch)
        logger.dumpkvs()

        if mean_r > best_score:
            best_score = mean_r
            torch.save(
                {
                    "epoch": epoch,
                    "theta_su": np.asarray(io_state.theta_su),
                    "norm_state": {
                        "mean": np.asarray(io_state.norm_state.mean),
                        "std": np.asarray(io_state.norm_state.std),
                    },
                    "augmenter": {
                        "rff_W": np.asarray(augmenter.rff_W),
                        "rff_b": np.asarray(augmenter.rff_b),
                        "rbf_centers": np.asarray(augmenter.rbf_centers),
                        "rbf_gamma": augmenter.rbf_gamma,
                        "scale_factor": augmenter.scale_factor,
                        "state_low": np.asarray(augmenter.state_low),
                        "state_high": np.asarray(augmenter.state_high),
                        "n_past": augmenter.n_past,
                        "use_rbf": augmenter.use_rbf,
                        "norm_momentum": augmenter.norm_momentum,
                    },
                },
                os.path.join(logger.checkpoint_dir, "policy.pth"),
            )

        print(f"Epoch {epoch + 1}: loss={loss:.4f}, reward={mean_r:.1f}+/-{std_r:.1f}")

    logger.log(f"Total: {time.time() - init_time:.1f}s")
    logger.close()


if __name__ == "__main__":
    import argparse
    import datetime
    from io_agent.plant.mujoco import HopperEnv, HopperExpertEnv, HopperMediumExpertEnv, Walker2dEnv, Walker2dExpertEnv, Walker2dMediumExpertEnv, HalfCheetahEnv

    ENVS = {"hopper": HopperEnv, "hopper-expert": HopperExpertEnv, "hopper-medium-expert": HopperMediumExpertEnv, "walker2d": Walker2dEnv, "walker2d-expert": Walker2dExpertEnv, "walker2d-medium-expert": Walker2dMediumExpertEnv, "cheetah": HalfCheetahEnv}

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="fast_io_test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--datasize", type=int, default=5000)
    parser.add_argument("--env-name", type=str, default="hopper")
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--step-per-epoch", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=5e-2)
    parser.add_argument("--lr-exp-decay", type=float, default=0.975)
    parser.add_argument("--n-past", type=int, default=4)
    parser.add_argument("--rff-n-features", type=int, default=1024)
    parser.add_argument("--rff-sigma", type=float, default=0.1)
    parser.add_argument("--no-rff", action="store_true", help="Disable RFF features")
    parser.add_argument("--no-co-product", action="store_true", help="Disable co_product features")
    parser.add_argument("--no-sinusoidal", action="store_true", help="Disable sinusoidal features")
    parser.add_argument("--sin-freqs", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0], help="Sinusoidal frequencies")
    parser.add_argument("--use-rbf", action="store_true", help="Enable RBF features")
    parser.add_argument("--rbf-n-features", type=int, default=1024)
    parser.add_argument("--rbf-sigma", type=float, default=0.1)
    parser.add_argument("--norm-momentum", type=float, default=0.0)
    parser.add_argument("--use-adam", action="store_true", help="Use Adam optimizer instead of SGD")
    parser.add_argument("--top-return-pct", type=float, default=1.0, help="Use top X%% trajectories by return (e.g., 0.1 for top 10%%)")
    parser.add_argument("--eval-episodes", type=int, default=40)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data-seed", type=int, default=None, help="Seed for random data sampling (default: use first N samples)")
    args = parser.parse_args()

    env = ENVS[args.env_name]()
    fast_args = FastIOArgs(
        seed=args.seed,
        obs_shape=env.observation_space.shape,
        action_dim=env.action_space.shape[0],
        learning_rate=args.learning_rate,
        lr_exp_decay=args.lr_exp_decay,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        step_per_epoch=args.step_per_epoch,
        datasize=args.datasize,
        epoch=args.epoch,
        eval_episodes=args.eval_episodes,
        n_past=args.n_past,
        rff_n_features=args.rff_n_features,
        rff_sigma=args.rff_sigma,
        use_rff=not args.no_rff,
        use_co_product=not args.no_co_product,
        use_sinusoidal=not args.no_sinusoidal,
        sin_freqs=tuple(args.sin_freqs),
        use_rbf=args.use_rbf,
        rbf_n_features=args.rbf_n_features,
        rbf_sigma=args.rbf_sigma,
        norm_momentum=args.norm_momentum,
        use_adam=args.use_adam,
        top_return_pct=args.top_return_pct,
        device=args.device,
        data_seed=args.data_seed,
    )

    timestamp = datetime.datetime.now().strftime("%y-%m%d-%H%M%S")
    log_dir = f"./logs/{args.exp_name}/fast_io/{args.env_name}/size_{args.datasize}/seed_{args.seed}/{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    logger = Logger(
        log_dir,
        {
            "consoleout_backup": "stdout",
            "policy_training_progress": "csv",
            "tb": "tensorboard",
        },
    )
    logger.log_hyperparameters(vars(fast_args))
    fast_io_trainer(fast_args, env, logger)
