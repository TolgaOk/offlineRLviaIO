from typing import Any, Tuple, Dict, Optional, Union
from warnings import warn
import numpy as np
import sympy
from gymnasium import spaces
import scipy


from io_agent.plant.base import (Plant,
                                 QuadraticCosts,
                                 LinearConstraint,
                                 LinearConstraints,
                                 InputValues,
                                 SystemInput,
                                 DynamicalSystem,
                                 LinearDiscreteSystem,
                                 Disturbances,
                                 NominalLinearEnvParams)


cont_a_matrix = sympy.Matrix([
    [-0.0226, -36.6170, -18.8970, -32.0900, 3.2509, -0.7626],
    [0.0001, -1.8997, 0.9831, -0.0007, -0.1708, -0.0050],
    [0.0123, 11.7200, -2.6316, 0.0009, -31.6040, 22.3960],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, -30, 0],
    [0, 0, 0, 0, 0, -30],
])
cont_b_matrix = sympy.Matrix([
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [30, 0],
    [0, 30],
])
disc_e_matrix = sympy.Matrix([
    [0, 0],
    [0, 0],
    [1, 0],
    [0, 1],
    [0, 0],
    [0, 0],
])


def make_continous_e_matrix(cont_a_matrix: sympy.Matrix,
                            disc_e_matrix: sympy.Matrix
                            ) -> sympy.Matrix:
    m = np.zeros((6 + 6, 6 + 6))
    m[:6, :6] = cont_a_matrix
    m[:6, 6:] = np.eye(6)
    exp_matrix = scipy.linalg.expm(m * 0.035)
    disc_lambda = exp_matrix[:6, 6:]

    return np.linalg.inv(disc_lambda) @ disc_e_matrix


cont_e_matrix = make_continous_e_matrix(cont_a_matrix, disc_e_matrix)
cont_c_matrix = sympy.eye(6)
cont_d_matrix = sympy.zeros(6, 2)

costs = QuadraticCosts(
    state=np.diag([1, 1000,  100, 1000, 1, 1]),
    action=np.eye(2),
    final=np.diag([1, 1000,  100, 1000, 1, 1]),
)


constraints = LinearConstraints(
    state=LinearConstraint(
        matrix=np.array(
            [[1, 0, 0, 0, 0, 0],
             [-1, 0, 0, 0, 0, 0]]),
        vector=np.ones((2)),
    ),
    action=LinearConstraint(
        matrix=np.array([
            [1, 0],
            [-1, 0],
            [0, 1],
            [0, -1],
        ]),
        vector=np.array([2., 2., 3., 3.])
    )
)


class FighterEnv(Plant):
    """ Linear fighter jet gym environment (Safonov et al., 1981).

    Args:
        max_length (int): Number of simulation step of the environment.
        env_params (DiscreteLinearEnvParams): Parameters that determines the behavior of the environment.
    """
    metadata = {"render_modes": []}

    def __init__(self,
                 max_length: int,
                 disturbance_bias: Optional[np.ndarray] = None,
                 ) -> None:
        self.sigma_v = np.array([
            [0.01, 0],
            [0, 0.001]
        ])
        super().__init__(costs=costs,
                         constraints=constraints,
                         disturbance_bias=Disturbances(
                             state=disturbance_bias,
                         ),
                         state_size=6,
                         action_size=2,
                         noise_size=2,
                         output_size=6,
                         max_length=max_length)

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.state_size,), dtype=float)
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(self.action_size,), dtype=float)

        self.dt = 0.035
        self.state = None
        self.iteration = None

        # Since Fighter is a linear model we only discretize the system once
        affine_sys = self.affinization(self.symbolic_dynamical_system())
        disc_affine_model = self.discretization(
            affine_sys=affine_sys,
            lin_point=InputValues(
                state=np.zeros((6,)),
                action=np.zeros((2,)),
                noise=np.zeros((2,)),
            )
        )
        self._nominal_model = NominalLinearEnvParams(
            matrices=LinearDiscreteSystem(
                a_matrix=disc_affine_model.a_matrix,
                b_matrix=disc_affine_model.b_matrix,
                c_matrix=disc_affine_model.c_matrix,
                d_matrix=disc_affine_model.d_matrix,
                e_matrix=disc_affine_model.e_matrix),
            constraints=self.constraints,
            costs=self.costs
        )

    def nominal_model(self, lin_point: Optional[InputValues] = None):
        return self._nominal_model

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
            dyn_eq=cont_a_matrix @ state
            + cont_b_matrix @ action
            + cont_e_matrix @ noise,
            out_eq=(cont_c_matrix @ state + cont_d_matrix @ action)
        )

    def fill_symbols(self,
                     input_values: InputValues,
                     ) -> Dict[str, Union[float, np.ndarray]]:
        values = dict(
            **{f"x_{index+1}": value for index, value in enumerate(input_values.state.flatten())},
            **{f"u_{index+1}": value for index, value in enumerate(input_values.action.flatten())},
            **{f"w_{index+1}": value for index, value in enumerate(input_values.noise.flatten())},
            dt=self.dt,
        )
        return values

    def generate_disturbance(self, rng: np.random.Generator) -> Disturbances:
        return Disturbances(
            state=np.stack([
                0.5 * np.sin(np.linspace(0, 6*np.pi, self.max_length * 2) + np.pi/2 * rng.random()),
                0.01 * np.ones(self.max_length * 2)],
                axis=0) + self.sigma_v @ rng.normal(size=(2, self.max_length * 2)),
            output=None,
            action=None,
        )

    def _measure(self) -> np.ndarray:
        """ Step output of the plant (does not use d_matrix)

        Returns:
            np.ndarray: Output array of shape (S,)
                where S denotes the state/input size
        """
        return (self._nominal_model.matrices.c_matrix @ self.state +
                self.disturbances.output[:, self.iteration])

    def _reset(self,
               rng: np.random.Generator,
               options: Optional[Dict[str, Any]] = None
               ) -> Tuple[Union[np.ndarray, Optional[Dict[str, Disturbances]]]]:
        """ Initialize the environment with random initial state
        """
        self.iteration = 0
        self.state = rng.random(self.state_size) / 10

        return self._measure(), dict()

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

        noisy_action = action + self.disturbances.action[:, self.iteration]
        state_noise = self.disturbances.state[:, self.iteration]
        reference = self.reference_sequence[:, self.iteration]

        next_state = (self._nominal_model.matrices.a_matrix @ self.state
                      + self._nominal_model.matrices.b_matrix @ noisy_action
                      + self._nominal_model.matrices.e_matrix @ state_noise)
        self.state = next_state
        measurement = self._measure()
        difference = (measurement - reference)
        cost = (difference.T.dot(self._nominal_model.costs.state).dot(difference)
                + noisy_action.T.dot(self._nominal_model.costs.action).dot(noisy_action))

        self.iteration += 1
        truncation = self.iteration == self.max_length
        terminal = False
        info = None
        return measurement, cost, truncation, terminal, info
