from typing import Any, Tuple, Dict, Optional, Union
import numpy as np
from gymnasium import spaces

from io_agent.plant.base import LinearEnvParams, Plant


fighter_env_params = LinearEnvParams(
    a_matrix=np.array([
        [0.9991, -1.3736, -0.6730, -1.1226, 0.3420, -0.2069],
        [0.0000, 0.9422, 0.0319, -0.0000, -0.0166, 0.0091],
        [0.0004, 0.3795, 0.9184, -0.0002, -0.6518, 0.4612],
        [0.0000, 0.0068, 0.0335, 1.0000, -0.0136, 0.0096],
        [0, 0, 0, 0, 0.3499, 0],
        [0, 0, 0, 0, 0, 0.3499],
    ]),
    b_matrix=np.array([
        [0.1457, -0.0819],
        [-0.0072, 0.0035],
        [-0.4085, 0.2893],
        [-0.0052, 0.0037],
        [0.6501, 0],
        [0, 0.6501],
    ]),
    e_matrix=np.array([
        [0, 0],
        [0, 0],
        [1, 0],
        [0, 1],
        [0, 0],
        [0, 0],
    ]),
    c_matrix=np.eye(6),
    state_cost=np.diag([1, 1000,  100, 1000, 1, 1]),
    action_cost=np.eye(2),
    final_cost=np.diag([1, 1000,  100, 1000, 1, 1]),
    state_constraint_matrix=np.array(
        [[1, 0, 0, 0, 0, 0],
         [-1, 0, 0, 0, 0, 0]]
    ),
    state_constraint_vector=np.ones((2)),
    action_constraint_matrix=np.array([
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1],
    ]),
    action_constraint_vector=np.array([2., 2., 3., 3.])
)


class FighterEnv(Plant):
    """ Linear fighter jet gym environment (Safonov et al., 1981).

    Args:
        max_length (int): Number of simulation step of the environment.
        env_params (LinearEnvParams): Parameters that determines the behavior of the environment.
    """

    metadata = {"render_modes": []}

    def __init__(self,
                  max_length: int,
                  env_params: LinearEnvParams,
                  disturbance_bias: Optional[np.ndarray]
                  ) -> None:
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6,), dtype=float)
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(2,), dtype=float)

        self.max_length = max_length
        self.sigma_v = np.array([
            [0.01, 0],
            [0, 0.001]
        ])

        self.state = None
        self.iteration = None
        self.state_disturbance = self._generate_state_disturbance()
        if disturbance_bias is not None:
            self.state_disturbance += disturbance_bias
        self.output_disturbance = np.zeros(
            (self.observation_space.shape[0], self.max_length * 2 + 1))
        self.action_disturbance = np.zeros((self.action_space.shape[0], self.max_length * 2))
        reference_sequence = np.zeros((self.observation_space.shape[0], self.max_length * 2))
        super().__init__(params=env_params, reference_sequence=reference_sequence)

    def _generate_state_disturbance(self) -> np.ndarray:
        """ Generate random time varying noise

        Returns:
            np.ndarray: Noise signal of shape (2, L) 
                where L denotes the environment length
        """
        return np.stack([
            0.5 * np.sin(np.linspace(0, 6*np.pi, self.max_length * 2) + np.pi/2 * np.random.rand()),
            0.01 * np.ones(self.max_length * 2)],
          axis=0) + self.sigma_v @ np.random.randn(2, self.max_length * 2)

    def _measure(self) -> np.ndarray:
        """ Step output of the plant

        Returns:
            np.ndarray: Output array of shape (S,)
                where S denotes the state/input size
        """
        return self.params.c_matrix @ self.state + self.output_disturbance[:, self.iteration]

    def reset(self,
              seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None
              ) -> Tuple[Union[np.ndarray, Optional[Dict[str, Any]]]]:
        """ Initialize the environment with random initial state

        Args:
            seed (Optional[int], optional): Random seed of the episode/trajectory. Defaults to random integer.
            options (Optional[Dict[str, Any]], optional): Options for the episode (unused). Defaults to None.

        Returns:
            Tuple[Union[np.ndarray, float, bool, Optional[Dict[str, Any]]]]:
                - initial output/state (np.ndarray): Array of shape (S,)
                     where S denotes the state/input size
                - info (Dict[str, Any]): Metadata of the initial state (set to None)
        """
        super().reset(seed=seed)
        self.iteration = 0
        self.state = np.random.rand(self.observation_space.shape[0]) / 10
        info = None
        return self._measure(), info

    def step(self,
             action: np.ndarray
             ) -> Tuple[Union[np.ndarray, float, bool, Optional[Dict[str, Any]]]]:
        """ One step transition function of the environment.

        Args:
            action (np.ndarray): Action/Input vector of shape (A,)
                where A denotes action space size

        Raises:
            RuntimeError: If the episode/trajectory has reached to maximum allowed step

        Returns:
            Tuple[Union[np.ndarray, float, bool, Optional[Dict[str, Any]]]]:
                - next output/state (np.ndarray) of shape (S,)
                     where S denotes the state/input size
                - cost (float): Transition cost
                - truncation (bool): Whether environment is terminated due to step limit
                - terminal (bool): Whether environment is naturally terminated
                - info (Dict[str, Any]): Metadata of the initial state (set to None)
        """
        if self.iteration >= self.max_length:
            raise RuntimeError("Environment is terminated. Call reset function first.")

        noisy_action = action + self.action_disturbance[:, self.iteration]
        state_noise = self.state_disturbance[:, self.iteration]
        reference = self.reference_sequence[:, self.iteration]

        next_state = self.params.a_matrix @ self.state + self.params.b_matrix @ noisy_action + \
            self.params.e_matrix @ state_noise
        difference = (self.state - reference)
        cost = (difference.T.dot(self.params.state_cost).dot(difference)
                + noisy_action.T.dot(self.params.action_cost).dot(noisy_action))
        self.iteration += 1
        truncation = self.iteration == self.max_length
        terminal = False
        info = None

        self.state = next_state

        return self._measure(), cost, truncation, terminal, info
