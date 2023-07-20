from typing import Any, Tuple, Dict, Optional, Union
from warnings import warn
from dataclasses import asdict
import numpy as np
import sympy
from gymnasium import spaces

from io_agent.plant.base import (Plant,
                                 LinearSystem,
                                 QuadraticCosts,
                                 LinearConstraints,
                                 DynamicalSystem,
                                 SystemInput,
                                 InputValues,
                                 EnvMatrices,
                                 DiscreteLinearEnvMatrices)


fighter_system = LinearSystem(
    a_matrix=sympy.Matrix([
        [-0.0226, -36.6170, -18.8970, -32.0900, 3.2509, -0.7626],
        [0.0001, -1.8997, 0.9831, -0.0007, -0.1708, -0.0050],
        [0.0123, 11.7200, -2.6316, 0.0009, -31.6040, 22.3960],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, -30, 0],
        [0, 0, 0, 0, 0, -30],
    ]),
    b_matrix=sympy.Matrix([
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [30, 0],
        [0, 30],
    ]),
    e_matrix=sympy.Matrix([
        [0, 0],
        [0, 0],
        [1, 0],
        [0, 1],
        [0, 0],
        [0, 0],
    ]),
    c_matrix=sympy.eye(6),
    d_matrix=sympy.zeros(6, 2)
)

costs = QuadraticCosts(
    state=np.diag([1, 1000,  100, 1000, 1, 1]),
    action=np.eye(2),
    final=np.diag([1, 1000,  100, 1000, 1, 1]),
)

constraints = LinearConstraints(
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
        env_params (DiscreteLinearEnvParams): Parameters that determines the behavior of the environment.
    """
    state_size: int = 6
    action_size: int = 2
    noise_size: int = 2
    output_size: int = 6
    metadata = {"render_modes": []}

    def __init__(self,
                 max_length: int,
                 disturbance_bias: Optional[np.ndarray] = None,
                 rng: Optional[np.random.Generator] = None,
                 ) -> None:
        self.n_state = 6
        self.n_action = 2
        self.n_noise = 2
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.n_state,), dtype=float)
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(self.n_action,), dtype=float)
        if rng is None:
            rng = np.random.default_rng()
            warn("Setting a random seed")
        self.rng = rng

        self.max_length = max_length
        self.sigma_v = np.array([
            [0.01, 0],
            [0, 0.001]
        ])
        self.dt = 0.035

        self.state = None
        self.iteration = None
        self.state_disturbance = self._generate_state_disturbance(self.rng)
        if disturbance_bias is not None:
            self.state_disturbance += disturbance_bias
        self.output_disturbance = np.zeros(
            (self.n_state, self.max_length * 2 + 1))
        self.action_disturbance = np.zeros((self.n_action, self.max_length * 2))
        reference_sequence = np.zeros((self.n_state, self.max_length * 2))
        super().__init__(costs=costs, constraints=constraints, reference_sequence=reference_sequence)

        discerete_system = self.discretize(
            fighter_system,
            lin_point=InputValues(
                state=np.zeros((6,)),
                action=np.zeros((2,)),
                noise=np.zeros((2,)),
            ),
            method="exact")
        self.env_params = EnvMatrices(
            **asdict(discerete_system),
            **asdict(constraints),
            state_cost=costs.state,
            action_cost=costs.action,
            final_cost=costs.final,
        )

    def symbolic_dynamical_system(self) -> DynamicalSystem:

        state = sympy.Matrix(sympy.symbols(" ".join([f"x_{index+1}" for index in range(6)])))
        action = sympy.Matrix(sympy.symbols(" ".join([f"u_{index+1}" for index in range(2)])))
        noise = sympy.Matrix(sympy.symbols(" ".join([f"w_{index+1}" for index in range(2)])))
        return DynamicalSystem(
            sys_input=SystemInput(
                state=state,
                action=action,
                noise=noise,
            ),
            dyn_eq=(fighter_system.a_matrix @ state
                    + fighter_system.b_matrix @ action
                    + fighter_system.e_matrix @ noise),
            out_eq=(fighter_system.c_matrix @ state + fighter_system.d_matrix @ action)
        )

    def fill_symbols(self, input_values: InputValues) -> Dict[str, Union[float, np.ndarray]]:
        return dict(
            **{f"x_{index+1}": value for index, value in enumerate(input_values.state.flatten())},
            **{f"u_{index+1}": value for index, value in enumerate(input_values.action.flatten())},
            **{f"w_{index+1}": value for index, value in enumerate(input_values.noise.flatten())},
            dt=0.035,
        )

    def _generate_state_disturbance(self, rng: np.random.Generator) -> np.ndarray:
        """ Generate random time varying noise

        Returns:
            np.ndarray: Noise signal of shape (2, L) 
                where L denotes the environment length
        """
        return np.stack([
            0.5 * np.sin(np.linspace(0, 6*np.pi, self.max_length * 2) + np.pi/2 * rng.random()),
            0.01 * np.ones(self.max_length * 2)],
            axis=0) + self.sigma_v @ rng.normal(size=(2, self.max_length * 2))

    def _measure(self) -> np.ndarray:
        """ Step output of the plant (does not use d_matrix)

        Returns:
            np.ndarray: Output array of shape (S,)
                where S denotes the state/input size
        """
        return self.env_params.c_matrix @ self.state + self.output_disturbance[:, self.iteration]

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
        rng = np.random.default_rng(seed)
        self.state = rng.random(self.n_state) / 10
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

        next_state = self.env_params.a_matrix @ self.state + self.env_params.b_matrix @ noisy_action + \
            self.env_params.e_matrix @ state_noise
        difference = (self.state - reference)
        cost = (difference.T.dot(self.env_params.state_cost).dot(difference)
                + noisy_action.T.dot(self.env_params.action_cost).dot(noisy_action))
        self.iteration += 1
        truncation = self.iteration == self.max_length
        terminal = False
        info = None

        self.state = next_state

        return self._measure(), cost, truncation, terminal, info
