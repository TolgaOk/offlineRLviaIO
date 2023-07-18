from typing import Any, Tuple, Dict, Optional, Union, List
import numpy as np
from scipy.linalg import sqrtm
import cvxpy as cp
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, asdict
from itertools import chain

from io_agent.evaluator import Transition
from io_agent.plant.base import EnvMatrices


@dataclass
class AugmentedTransition(Transition):
    aug_state: np.ndarray
    expert_action: np.ndarray
    constraint_matrix: Optional[np.ndarray] = None
    constraint_vector: Optional[np.ndarray] = None
    hindsight_weight: Optional[np.ndarray] = None


class FeatureHandler():

    def __init__(self,
                 env_params: EnvMatrices,
                 n_past: int,
                 add_bias: bool,
                 use_state_regressor: bool,
                 use_action_regressor: bool,
                 use_noise_regressor: bool,
                 ) -> None:
        """ Feature Handler for the IO agents

        Args:
            env_params (EnvMatrices): Parameters that determines the behavior of the environment.
            n_past (int): The history length of the augmented states
            add_bias (bool): Add bias to the (state features/augmented state)
            use_state_regressor (bool): Include past states to the (state features/augmented state)
            use_action_regressor (bool): Include past actions to the (state features/augmented state)
            use_noise_regressor (bool): Include past noises to the (state features/augmented state)

        """
        self.env_params = env_params
        self.n_past = n_past
        self.add_bias = add_bias
        self.use_state_regressor = use_state_regressor
        self.use_action_regressor = use_action_regressor
        self.use_noise_regressor = use_noise_regressor

        self.noise_size = env_params.e_matrix.shape[1]
        self.state_size = env_params.a_matrix.shape[1]
        self.action_size = env_params.b_matrix.shape[1]

    @property
    def aug_state_size(self) -> int:
        """ Compute the augmented state/feature size

        Returns:
            int: Size of the augmented state/features 
        """
        return (
            self.state_size
            + int(self.add_bias)
            + (self.state_size * self.n_past) * int(self.use_state_regressor)
            + (self.noise_size * self.n_past) * int(self.use_noise_regressor)
            + (self.action_size * self.n_past) * int(self.use_action_regressor)
        )

    def reset_history(self) -> Dict[str, np.ndarray]:
        """ Reset the history of the state, noise and action lists

        Returns:
            Dict[str, np.ndarray]: Cleaned history
        """
        return dict(
            noise=np.zeros((self.n_past, self.noise_size)),
            state=np.zeros((self.n_past, self.state_size)),
            action=np.zeros((self.n_past, self.action_size)),
        )

    def infer_noise(self,
                    state: np.ndarray,
                    next_state: np.ndarray,
                    action: np.ndarray
                    ) -> np.ndarray:
        """ Infer the noise from the state transition

        Args:
            state (np.ndarray): State array of shape (S,) where S denotes the
                output/state space size
            next_state (np.ndarray): Next state array of shape (S,) where S 
                denotes the output/state space size
            action (np.ndarray): Action array of shape (A,) where A denotes the
                output/state space size

        Returns:
            np.ndarray: Inferred noise array of shape (W,) where W denotes
                the noise size
        """
        return np.linalg.pinv(self.env_params.e_matrix) @ (
            next_state
            - self.env_params.a_matrix @ state
            - self.env_params.b_matrix @ action)

    def update_history(self,
                       state: np.ndarray,
                       next_state: np.ndarray,
                       action: np.ndarray,
                       history: Dict[str, np.ndarray]
                       ) -> Dict[str, np.ndarray]:
        """ Update the given history with the transition arrays

        Args:
            state (np.ndarray): State array of shape (S,) where S denotes the
                output/state space size
            next_state (np.ndarray): Next state array of shape (S,) where S 
                denotes the output/state space size
            action (np.ndarray): Action array of shape (A,) where A denotes the
                output/state space size
            history (Dict[str, np.ndarray]): History dictionary

        Returns:
            Dict[str, np.ndarray]: Updated history
        """
        noise = self.infer_noise(
            state=state,
            next_state=next_state,
            action=action)
        for name, new_vector in (("noise", noise), ("state", state), ("action", action)):
            history[name][1:] = history[name][:-1]
            history[name][0] = new_vector
        return history

    def augment_state(self, state: np.ndarray, history: Dict[str, np.ndarray]) -> np.ndarray:
        """ Augment the state using the given history

        Args:
            state (np.ndarray): State array of shape (S,) where S denotes the
                output/state space size
            history (Dict[str, np.ndarray]): History dictionary

        Returns:
            np.ndarray: Augmented state/features
        """
        features = []
        for name, condition in (("noise", self.use_noise_regressor),
                                ("state", self.use_state_regressor),
                                ("action", self.use_action_regressor)):
            if condition:
                features.append(history[name].flatten())
        if self.add_bias:
            features.append(np.ones((1,)))
        return np.concatenate([state, *features])
