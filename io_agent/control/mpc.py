from typing import Any, Tuple, Dict, Optional, Union
import numpy as np
from scipy.linalg import sqrtm
import cvxpy as cp
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass

from io_agent.plant.base import LinearEnvParams


@dataclass
class Optimizer:
    problem: Any
    parameters: Any
    variables: Any


class MPC():
    """ Linear MPC agent

    Args:
        env_params (LinearEnvParams): Linear environment parameters
        horizon (int): MPC horizon
    """

    def __init__(self,
                 env_params: LinearEnvParams,
                 horizon: int,
                 ) -> None:
        self.env_params = env_params
        self.horizon = horizon

        self.action_size = self.env_params.b_matrix.shape[-1]
        self.state_size = self.env_params.a_matrix.shape[-1]
        self.noise_size = self.env_params.e_matrix.shape[-1]
        self.ref_size = self.env_params.a_matrix.shape[-1]
        self.output_size = self.env_params.a_matrix.shape[-1]

        self.optimizer = self.prepare_optimizer()

    def compute(self,
                initial_state: np.ndarray,
                reference_sequence: np.ndarray,
                output_disturbance: Optional[np.ndarray] = None,
                state_disturbance: Optional[np.ndarray] = None,
                ) -> Tuple[Union[np.ndarray, float]]:
        """ Compute the optimal actions (if possible) and total cost

        Args:
            initial_state (np.ndarray): Initial state of shape (S,)
                where S denotes the output/state space size
            reference_sequence (np.ndarray): Reference signal array to follow of shape (S, T)
                where L denotes the environment length
            output_disturbance (Optional[np.ndarray], optional): Output disturbance array of
                shape (S, T). Defaults to None.
            state_disturbance (Optional[np.ndarray], optional): State disturbance array of
                shape (W, T). Defaults to None. Where W denotes the length of the state
                disturbance dimension.

        Returns:
            Tuple[Union[np.ndarray, float, None]]:
                - Optimal action vector (if possible) of shape (A,) at the initial time
                    step where A denotes the action space size or None if the solver
                    cannot solve the problem.
                - Total cost of the MPC problem (float)
        """

        if output_disturbance is None:
            output_disturbance = np.zeros((self.ref_size, self.horizon))

        if state_disturbance is None:
            state_disturbance = np.zeros((self.noise_size, self.horizon))

        self.optimizer.parameters["initial_state"].value = initial_state
        self.optimizer.parameters["reference_sequence"].value = (
            reference_sequence - output_disturbance)
        self.optimizer.parameters["state_disturbance"].value = state_disturbance

        result = self.optimizer.problem.solve(solver=cp.MOSEK)
        return self.optimizer.variables["actions"].value[:, 0], result

    def prepare_optimizer(self) -> Optimizer:
        """ Prepare a parametric optimization problem for the mpc agent

        Returns:
            Optimizer: Optimization problem, parameters, and variables
        """
        action_list = []
        constraints = []
        cost = 0

        x_par = cp.Parameter(self.state_size)
        r_par = cp.Parameter((self.state_size, self.horizon))
        w_par = cp.Parameter((self.noise_size, self.horizon))

        state = x_par
        for step in range(self.horizon):
            action_var = cp.Variable((self.action_size), name=f"mu_{step+1}")
            action_list.append(action_var)
            state = (self.env_params.a_matrix @ state +
                     self.env_params.b_matrix @ action_var +
                     self.env_params.e_matrix @ w_par[:, step])
            state_cost = (self.env_params.state_cost if step < self.horizon - 1
                          else self.env_params.final_cost)
            cost = cost + cp.quad_form(state, state_cost)
            cost = cost + cp.quad_form(action_var, self.env_params.action_cost)

            constraints += [self.env_params.state_constraint_matrix @ state <=
                            self.env_params.state_constraint_vector]
            constraints += [self.env_params.action_constraint_matrix @ action_var <=
                            self.env_params.action_constraint_vector]
        objective = cp.Minimize(cost)
        problem = cp.Problem(objective, constraints)

        return Optimizer(
            problem=problem,
            parameters={
                "initial_state": x_par,
                "reference_sequence": r_par,
                "state_disturbance": w_par
            },
            variables={"actions": cp.hstack(
                [cp.reshape(action, (self.action_size, 1)) for action in action_list])}
        )
