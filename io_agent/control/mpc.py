from typing import Any, Tuple, Dict, Optional, Union
import numpy as np
from scipy.linalg import sqrtm
import cvxpy as cp
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass

from io_agent.plant.base import NominalLinearEnvParams


@dataclass
class Optimizer:
    problem: Any
    parameters: Any
    variables: Any


class MPC():
    """ Linear MPC agent

    Args:
        horizon (int): MPC horizon
    """

    def __init__(self,
                 action_size: int,
                 state_size: int,
                 noise_size: int,
                 output_size: int,
                 horizon: int,
                 ) -> None:

        self.action_size = action_size
        self.state_size = state_size
        self.noise_size = noise_size
        self.output_size = output_size
        self.horizon = horizon

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
            output_disturbance = np.zeros((self.output_size, self.horizon))

        if state_disturbance is None:
            state_disturbance = np.zeros((self.noise_size, self.horizon))
        self.optimizer.parameters["initial_state"].value = initial_state
        self.optimizer.parameters["reference_sequence"].value = (
            reference_sequence - output_disturbance)
        self.optimizer.parameters["state_disturbance"].value = state_disturbance

        result = self.optimizer.problem.solve(solver=cp.MOSEK)
        return self.optimizer.variables["actions"].value[:, 0], result

    def reset(self) -> None:
        pass

    def prepare_optimizer(self, params: NominalLinearEnvParams) -> Optimizer:
        """ Prepare a parametric optimization problem for the mpc agent

        Returns:
            Optimizer: Optimization problem, parameters, and variables
        """
        action_list = []
        constraints = []
        cost = 0

        init_state = cp.Parameter(self.state_size)
        r_par = cp.Parameter((self.state_size, self.horizon))
        w_par = cp.Parameter((self.noise_size, self.horizon))

        lin_state = params.matrices.lin_input.state  # Linearization point of the state
        lin_next_state = params.matrices.nonlinear_dyn  # Linearization point of the next_state
        lin_action = params.matrices.lin_input.action  # Linearization point of the action
        lin_noise = params.matrices.lin_input.noise  # Linearization point of the noise
        lin_output = params.matrices.nonlinear_out  # Linearization point of the output

        state_delta = init_state - lin_state
        for step in range(self.horizon):
            action_var = cp.Variable((self.action_size), name=f"mu_{step+1}")
            action_list.append(action_var)
            state_delta = (params.matrices.a_matrix @ (state_delta) +
                           params.matrices.b_matrix @ (action_var - lin_action) +
                           params.matrices.e_matrix @ (w_par[:, step] - lin_noise) +
                           lin_next_state)
            state_cost = (params.costs.state if step < self.horizon - 1
                          else params.costs.final)
            cost = cost + cp.quad_form((state_delta + lin_state), state_cost)
            cost = cost + cp.quad_form(action_var, params.costs.action)

            constraints += [params.constraints.state_constraint_matrix @ (state_delta + lin_state) <=
                            params.constraints.state_constraint_vector]
            constraints += [params.constraints.action_constraint_matrix @ action_var <=
                            params.constraints.action_constraint_vector]
        objective = cp.Minimize(cost)
        problem = cp.Problem(objective, constraints)

        return Optimizer(
            problem=problem,
            parameters={
                "initial_state": init_state,
                "reference_sequence": r_par,
                "state_disturbance": w_par
            },
            variables={"actions": cp.hstack(
                [cp.reshape(action, (self.action_size, 1)) for action in action_list])}
        )
