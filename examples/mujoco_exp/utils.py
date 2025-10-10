from typing import List, Tuple, Dict
import numpy as np

from io_agent.plant.mujoco import MuJoCoEnv
from io_agent.evaluator import Transition
from io_agent.control.io import FeatureHandler, AugmentDataset
from io_agent.utils import save_experiment


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
    rewards = dataset["rewards"]

    dones = np.argwhere(np.logical_or(
        dataset["terminals"], dataset["timeouts"]))

    start_index = 0
    for done_index in dones:
        done_index = done_index.item()
        trajectory = []
        episode_return = rewards[start_index:done_index + 1].sum()
        episode_length = done_index + 1 - start_index
        for index in range(start_index, done_index + 1):
            trajectory.append(
                Transition(
                    state=states[index],
                    action=actions[index],
                    next_state=None,
                    cost=None,
                    termination=None,
                    truncation=None,
                    info={
                        "episode_return": episode_return,
                        "episode_length": episode_length,
                    },
                    reference=None,
                )
            )
        start_index = done_index + 1
        trajectories.append(trajectory)

    # Sanity check
    start_indices = np.concatenate(
        [np.zeros((1,), dtype=np.int32), dones.ravel()[:-1].astype(np.int32) + 1])
    traj_lengths = (dones + 1).ravel() - start_indices
    assert np.all([len(traj) == traj_lengths[index]
                  for index, traj in enumerate(trajectories)])

    augmented_trajectories = augmenter(trajectories)
    save_experiment(
        values={
            "augmented_dataset": augmented_trajectories,
            "feature_handler": feature_handler},
        seed=None,  # The process is deterministic
        exp_dir=save_dir,
        name=file_name
    )
