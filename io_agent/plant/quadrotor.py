from typing import Any, Tuple, Dict, Optional, Union
import numpy as np
import sympy

from safe_control_gym.utils.registration import make
from safe_control_gym.envs.disturbances import Disturbance, DisturbanceList

from io_agent.plant.base import (Plant,
                                 QuadraticCosts,
                                 LinearConstraints,
                                 LinearConstraint,
                                 InputValues,
                                 SystemInput,
                                 DynamicalSystem,
                                 Disturbances)


task_config = {
    "info_in_reset": False,
    "ctrl_freq": 60,
    "pyb_freq": 240,
    "physics": "pyb",
    "gui": False,
    "quad_type": 2,
    "normalized_rl_action_space": False,
    "episode_len_sec": 5,
    "init_state": None,
    "randomized_init": True,
    "init_state_randomization_info": None,
    "inertial_prop": None,
    "randomized_inertial_prop": False,
    "inertial_prop_randomization_info": None,
    "task": "stabilization",
    "task_info": None,
    "cost": "quadratic",
    "disturbances": {},
    "adversary_disturbance": None,
    "adversary_disturbance_offset": 0.0,
    "adversary_disturbance_scale": 0.01,
    "constraints": None,
    "done_on_violation": False,
    "use_constraint_penalty": False,
    "constraint_penalty": -1,
    "verbose": False,
    "norm_act_scale": 0.1,
    "obs_goal_horizon": 0,
    "rew_state_weight": 1.0,
    "rew_act_weight": 0.0001,
    "rew_exponential": True,
    "done_on_out_of_bound": True
}


class PeriodicExternalForceDisturbance(Disturbance):

    def __init__(self,
                 env,
                 dim,
                 max_length: int,
                 rng: np.random.Generator,
                 ratio: float = 0.1
                 ):
        super().__init__(env, dim)

        self.max_length = max_length
        scale = (env.action_space.high - env.action_space.low).reshape(-1, 1) * ratio
        self.disturbance = scale * (
            (np.sin(np.linspace(0, 6*np.pi, self.max_length * 2) + np.pi/2 * rng.random(size=(dim, 1)))
             + rng.uniform(-0.5, 0.5, (dim, self.max_length * 2)))
        )

    def apply(self,
              target,
              env
              ):
        index = env.ctrl_step_counter
        return target + self.disturbance[:, index]


class QuadrotorEnv(Plant):

    def __init__(self, use_exp_reward: bool = False) -> None:

        self.env = make("quadrotor", **task_config)

        self.reference_state = self.env.X_GOAL
        self.use_exp_reward = use_exp_reward
        # self.reference_action = self.env.U_GOAL
        self.dyn_sys = self.symbolic_dynamical_system()
        self._state = None

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        constraint_indices = self.env.observation_space.high < 1e+10

        super().__init__(
            state_size=self.observation_space.shape[0],
            action_size=self.action_space.shape[0],
            noise_size=self.env.DISTURBANCE_MODES["dynamics"]["dim"],
            output_size=self.observation_space.shape[0],
            max_length=self.env.CTRL_STEPS,
            costs=QuadraticCosts(
                state=self.env.Q * 0.5,
                action=self.env.R * 0.5,
                final=self.env.Q * 0.5
            ),
            constraints=LinearConstraints(
                state=LinearConstraint(
                    matrix=np.block([
                        [np.eye(self.env.observation_space.shape[0])[constraint_indices]],
                        [-np.eye(self.env.observation_space.shape[0])[constraint_indices]]]),
                    vector=np.concatenate([
                        self.env.observation_space.high[constraint_indices],
                        -self.env.observation_space.low[constraint_indices]
                    ])
                ),
                action=LinearConstraint(
                    matrix=np.block([
                        [np.eye(self.env.action_space.shape[0])],
                        [-np.eye(self.env.action_space.shape[0])]]),
                    vector=np.concatenate([
                        self.env.action_space.high,
                        -self.env.action_space.low
                    ])
                )
            ),
            reference_sequence=(np.ones((1, self.env.CTRL_STEPS * 2))
                                * self.reference_state.reshape(-1, 1)),
            disturbance_bias=None,
        )

    def symbolic_dynamical_system(self) -> DynamicalSystem:
        m, g, l, i_yy = sympy.symbols("m g l I_{yy}")

        x, x_dot, z, z_dot, theta, theta_dot = sympy.symbols(
            "x \dot{x} z \dot{z} \\theta \dot{\\theta}")
        u_1, u_2 = sympy.symbols("u_1 u_2")
        w_x, w_z = sympy.symbols("w_x w_z")

        state = sympy.Matrix([x, x_dot, z, z_dot, theta, theta_dot])
        action = sympy.Matrix([u_1, u_2])
        state_noise = sympy.Matrix([w_x, w_z])

        dyn_eq = sympy.Matrix([
            x_dot,
            (sympy.sin(theta) * (u_1 + u_2) + w_x) / m,
            z_dot,
            (sympy.cos(theta) * (u_1 + u_2) + w_z) / m - g,
            theta_dot,
            l * (u_2 - u_1) / i_yy / sympy.sqrt(2)
        ])
        out_eq = sympy.Matrix([x, x_dot, z, z_dot, theta, theta_dot])
        return DynamicalSystem(
            sys_input=SystemInput(
                state=state,
                action=action,
                noise=state_noise,
            ),
            dyn_eq=dyn_eq,
            out_eq=out_eq
        )

    def fill_symbols(self,
                     input_values: InputValues,
                     ) -> Dict[str, Union[float, np.ndarray]]:
        state = input_values.state.flatten()
        action = input_values.action.flatten()
        noise = input_values.noise.flatten()
        values = {
            "x": state[0],
            "\dot{x}": state[1],
            "z": state[2],
            "\dot{z}": state[3],
            "\\theta": state[4],
            "\dot{\\theta}": state[5],
            "u_1": action[0],
            "u_2": action[1],
            "w_x": noise[0],
            "w_z": noise[1],
        }
        constants = {
            "m": self.env.MASS,
            "g": self.env.GRAVITY_ACC,
            "l": self.env.L,
            "I_{yy}": self.env.J[1, 1],
            "dt": self.env.symbolic.dt
        }
        values.update(constants)
        return values

    def generate_disturbance(self, rng: np.random.Generator) -> Disturbances:
        force_disturbance = PeriodicExternalForceDisturbance(
            env=self.env, dim=self.noise_size, max_length=self.max_length, rng=rng
        )
        self.env.disturbances["dynamics"] = DisturbanceList(disturbances=[force_disturbance])
        return Disturbances(
            state=force_disturbance.disturbance
        )

    def _reset(self,
               rng: Optional[int] = None,
               options: Optional[Dict[str, Any]] = None
               ) -> Tuple[Union[np.ndarray, Optional[Dict[str, Any]]]]:
        self._state = self.env.reset(seed=rng.integers(0, 2**30).item())
        return self._state, {}
    
    def default_lin_point(self) -> InputValues:
        return InputValues(
            state=self.env.symbolic.X_EQ,
            action=self.env.symbolic.U_EQ,
            noise=np.zeros((self.noise_size,)),
        )

    def step(self,
             action: np.ndarray
             ) -> Tuple[Union[np.ndarray, float, bool, Optional[Dict[str, Any]]]]:
        self._state, reward, done, info = self.env.step(action)
        cost = np.exp(reward).item() if self.use_exp_reward else -reward
        return self._state, cost, done, done, info
