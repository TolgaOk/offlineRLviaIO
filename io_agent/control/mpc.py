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
                action_disturbance: Optional[np.ndarray] = None,
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

    def cost_function(self,
                      params: NominalLinearEnvParams,
                      step: int,
                      reference: Any,
                      state: Any,
                      action: Any,
                      next_state: Any) -> Any:
        state_cost = (params.costs.state if step < self.horizon - 1
                      else params.costs.final)
        return (cp.quad_form((params.matrices.c_matrix @ next_state - reference), state_cost)
                + cp.quad_form(action, params.costs.action))

    def prepare_optimizer(self, params: NominalLinearEnvParams) -> Optimizer:
        """ Prepare a parametric optimization problem for the mpc agent

        Returns:
            Optimizer: Optimization problem, parameters, and variables
        """
        action_list = []
        constraints = []
        cost = 0

        if not np.allclose(params.matrices.d_matrix, 0, atol=1e-6):
            raise ValueError("D matrix is not supported. It must be all zeros.")

        init_state = cp.Parameter(self.state_size)
        r_par = cp.Parameter((self.output_size, self.horizon))
        w_par = cp.Parameter((self.noise_size, self.horizon))

        state = init_state
        for step in range(self.horizon):
            action_var = cp.Variable((self.action_size), name=f"mu_{step+1}")
            action_list.append(action_var)
            next_state = (params.matrices.a_matrix @ (state) +
                          params.matrices.b_matrix @ action_var +
                          params.matrices.e_matrix @ w_par[:, step])
            cost = cost + self.cost_function(
                params=params,
                step=step,
                reference=r_par[:, step],
                state=state,
                action=action_var,
                next_state=next_state)
            # state_cost = (params.costs.state if step < self.horizon - 1
            #               else params.costs.final)
            # cost = cost + cp.quad_form((params.matrices.c_matrix @ state - r_par[:, step]), state_cost)
            # cost = cost + cp.quad_form(action_var, params.costs.action)

            if params.constraints.state is not None:
                constraints += [params.constraints.state.matrix @ state <=
                                params.constraints.state.vector]
            if params.constraints.action is not None:
                constraints += [params.constraints.action.matrix @ action_var <=
                                params.constraints.action.vector]
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


class MuJoCoMPC(MPC):

    dt: float = 0.008
    alpha: float = 1e-3

    def cost_function(self,
                      params: NominalLinearEnvParams,
                      step: int,
                      reference: Any,
                      state: Any,
                      action: Any,
                      next_state: Any) -> Any:
        return self.alpha * cp.quad_form(action, np.eye(self.action_size)) \
            - (next_state[0] - state[0]) / self.dt \
            - 1
