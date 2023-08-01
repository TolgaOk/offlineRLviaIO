from typing import Any, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from functools import partial
import numpy as np
import sympy
from scipy.integrate import odeint
from gymnasium import spaces


from io_agent.plant.base import (Plant,
                                 QuadraticCosts,
                                 LinearConstraints,
                                 InputValues,
                                 SystemInput,
                                 DynamicalSystem,
                                 Disturbances,
                                 NominalLinearEnvParams)


costs = QuadraticCosts(
    state=np.eye(2),
    action=np.diag([0.1, 0.05]),
    final=np.eye(2)
)
constraints = LinearConstraints()


@dataclass
class Constants:
    alpha_1: float = 0.0109
    alpha_2: float = 0.0058
    cp: float = 500
    A: float = 1e-3
    As: float = 2e-4
    m: float = 0.004
    U: float = 4.0
    Us: float = 36.5
    epsilon: float = 0.9
    sigma: float = 5.67e-8
    tau: float = 18.3
    c2k: float = 273.15


class DualHeaterEnv(Plant):

    def __init__(self,
                 max_length: int,
                 disturbance_bias: Optional[np.ndarray] = None,
                 ) -> None:
        self.dt = 10
        self.ambient_temp = 23
        self.iteration = None
        self.reference_temp = np.array([55.0, 45.0])
        self.dyn_sys = self.symbolic_dynamical_system()

        self.observation_space = spaces.Box(
            low=np.ones((4,)) * (-np.inf),
            high=np.ones((4,)) * (np.inf)
        )
        self.action_space = spaces.Box(
            low=np.ones((2,)) * 0.0,
            high=np.ones((2,)) * 100.0
        )

        super().__init__(
            state_size=self.observation_space.shape[0],
            action_size=self.action_space.shape[0],
            noise_size=1,
            output_size=2,
            max_length=max_length,
            costs=costs,
            constraints=constraints,
            reference_sequence=np.ones((1, max_length * 2)) * self.reference_temp.reshape(-1, 1),
            disturbance_bias=disturbance_bias,
        )

    def symbolic_dynamical_system(self) -> DynamicalSystem:
        action = sympy.Matrix(sympy.symbols(" ".join([f"u_{index+1}" for index in range(2)])))
        state = sympy.Matrix(sympy.symbols(" ".join([f"x_{index+1}" for index in range(4)])))
        state_noise = sympy.Matrix([sympy.symbols("w_x")])
        output_noise = sympy.Matrix([sympy.symbols("w^o_1 w^o_2")])

        a1, a2, a3, a4, tau_h, tau_c, b1, b2, c2k = sympy.symbols(
            "a_1 a_2 a_3 a_4 \\tau_h \\tau_c b_1 b_2 T")

        ta = state_noise[0] + c2k

        ca1 = a1 * (ta - state[0])
        ca2 = a1 * (ta - state[1])
        ra1 = a2 * (ta ** 4 - state[0] ** 4)
        ra2 = a2 * (ta ** 4 - state[1] ** 4)
        c21 = a3 * (state[1] - state[0])
        r21 = a4 * (state[1] ** 4 - state[0] ** 4)

        state_dot = sympy.Matrix([
            (ca1 + ra1 + c21 + r21 + b1 * action[0]) / tau_h,
            (ca2 + ra2 - c21 - r21 + b2 * action[1]) / tau_h,
            (state[0] - state[2]) / tau_c,
            (state[1] - state[3]) / tau_c,
        ])
        return DynamicalSystem(
            sys_input=SystemInput(
                state=state,
                action=action,
                noise=state_noise,
                output_noise=output_noise
            ),
            dyn_eq=state_dot,
            out_eq=sympy.Matrix([
                state[2] - c2k + output_noise[0],
                state[3] - c2k + output_noise[1],
            ])
        )

    def fill_symbols(self,
                     input_values: InputValues,
                     ) -> Dict[str, Union[float, np.ndarray]]:
        values = {
            **{f"x_{index+1}": value for index, value in enumerate(input_values.state.flatten())},
            **{f"u_{index+1}": value for index, value in enumerate(input_values.action.flatten())},
            **{f"w^o_{index+1}": value for index, value in enumerate(input_values.output_noise.flatten())},
            f"w_x": input_values.noise[0],
            "dt": self.dt,
        }
        values.update(
            {"a_1": Constants.U * Constants.A,
             "a_2": Constants.epsilon * Constants.sigma * Constants.A,
             "a_3": Constants.As * Constants.Us,
             "a_4": Constants.epsilon * Constants.sigma * Constants.As,
             "\\tau_h": Constants.m * Constants.cp,
             "\\tau_c": Constants.tau,
             "b_1": Constants.alpha_1,
             "b_2": Constants.alpha_2,
             "T": Constants.c2k
             })
        return values

    def generate_disturbance(self, rng: np.random.Generator) -> Disturbances:
        return Disturbances(
            state=(rng.normal(size=(1, self.max_length * 2)) +
                   self.ambient_temp + 10 * rng.uniform() - 5)
        )

    def _reset(self,
               rng: Optional[int] = None,
               options: Optional[Dict[str, Any]] = None
               ) -> Tuple[Union[np.ndarray, Optional[Dict[str, Any]]]]:
        self.iteration = 0
        mean_init_state = ((self.ambient_temp - 5)
                        #    - np.concatenate([self.reference_temp, self.reference_temp]
                           ) + Constants.c2k
        self.state = mean_init_state + \
            rng.uniform(-5., 5., size=(1,)) + rng.normal(size=(self.state_size,))
        return self.state, {}

    def default_lin_point(self) -> InputValues:
        u_1 = sympy.solve(self.dyn_sys.dyn_eq[0], self.dyn_sys.sys_input.action[0])[0]
        u_2 = sympy.solve(self.dyn_sys.dyn_eq[1], self.dyn_sys.sys_input.action[1])[0]
        action = sympy.Matrix([u_1, u_2])

        init_lin_point = InputValues(
            state=np.concatenate([self.reference_temp,
                                  self.reference_temp]
                                 ) + Constants.c2k,
            action=np.zeros((self.action_size,)),
            noise=np.ones(self.noise_size,) * self.ambient_temp,
            output_noise=np.zeros((self.output_size,))
        )
        lin_action = self.evaluate_sym(
            action, values=self.fill_symbols(init_lin_point)
        ).flatten()
        return InputValues(
            state=init_lin_point.state,
            action=lin_action,
            noise=init_lin_point.noise,
            output_noise=init_lin_point.output_noise,
        )

    def _measure(self) -> np.ndarray:
        return self.state[2:] - Constants.c2k + self.disturbances.output[:, self.iteration]

    def _state_dot(self,
                   state: np.ndarray,
                   time: float,
                   action: np.ndarray,
                   dyn_eq: sympy.Matrix,
                   step: int):
        return self.evaluate_sym(dyn_eq, values=self.fill_symbols(InputValues(
            state=state,
            action=action,
            noise=self.disturbances.state[:, step],
            output_noise=np.zeros((self.output_size,))
        ))).flatten()

    def step(self,
             action: np.ndarray
             ) -> Tuple[Union[np.ndarray, float, bool, Optional[Dict[str, Any]]]]:

        noisy_action = action + self.disturbances.action[:, self.iteration]
        solution = odeint(func=partial(
            self._state_dot,
            action=noisy_action,
            dyn_eq=self.dyn_sys.dyn_eq,
            step=self.iteration),
            t=[0, self.dt],
            y0=self.state)
        self.state = solution[-1]
        self.iteration += 1
        truncation = self.iteration == self.max_length
        terminal = truncation

        output = self._measure()
        difference = (output - self.reference_temp)

        state_cost = self.costs.state if not terminal else self.costs.final
        cost = (difference.T.dot(state_cost).dot(difference)
                + noisy_action.T.dot(self.costs.action).dot(noisy_action))
        info = dict()
        return self.state, cost, truncation, terminal, info
