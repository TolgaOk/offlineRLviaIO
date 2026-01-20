"""Fast evaluation for fast_io controller."""

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from io_agent.control.fast_io import Augmenter, AugmenterState, AugNormState, KernelAugmenter, KernelAugmenterState, compute_action


def _make_eval_fns(augmenter):
    """Create JIT-compiled eval functions for given augmenter (Augmenter or KernelAugmenter)."""
    state_class = KernelAugmenterState if isinstance(augmenter, KernelAugmenter) else AugmenterState

    @jax.jit
    def augment_obs(obs, aug_state, norm_state):
        return augmenter._augment(obs, aug_state.history_obs, aug_state.history_act, norm_state)

    @jax.jit
    def update_history(aug_state, obs, action):
        new_history_obs = jnp.roll(aug_state.history_obs, 1, axis=0).at[0].set(obs)
        new_history_act = jnp.roll(aug_state.history_act, 1, axis=0).at[0].set(action)
        return state_class(history_obs=new_history_obs, history_act=new_history_act)

    @jax.jit
    def get_action(theta_su, aug_obs):
        return compute_action(theta_su, aug_obs)

    return augment_obs, update_history, get_action


def run_episode(theta_su, env, seed, augmenter, norm_state, augment_obs, update_history, get_action):
    """Run single episode with pre-compiled functions."""
    state, _ = env.reset(seed=seed)
    state = np.asarray(state)
    aug_state = augmenter.reset()

    total_reward = 0.0
    done = False

    while not done:
        aug_obs = augment_obs(jnp.asarray(state), aug_state, norm_state)
        action = np.asarray(get_action(theta_su, aug_obs))

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        aug_state = update_history(aug_state, jnp.asarray(state), jnp.asarray(action))
        state = np.asarray(next_state)
        total_reward += reward

    return total_reward


def evaluate(theta_su, env, seeds, augmenter: Augmenter, norm_state: AugNormState) -> np.ndarray:
    """Evaluate over multiple seeds."""
    augment_obs, update_history, get_action = _make_eval_fns(augmenter)
    return np.array([
        run_episode(theta_su, env, int(s), augmenter, norm_state, augment_obs, update_history, get_action)
        for s in seeds
    ])
