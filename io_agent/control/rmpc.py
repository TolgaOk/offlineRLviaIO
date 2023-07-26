from typing import Any, Tuple, Dict, Optional, Union
import numpy as np
from scipy.linalg import sqrtm
import cvxpy as cp

from io_agent.plant.base import NominalLinearEnvParams
from io_agent.control.mpc import Optimizer, MPC


class RobustMPC(MPC):
    """ Robust Linear MPC agent

    Args:
        horizon (int): MPC horizon
        rho (float): Disturbance uncertainty radius
        p_matrix (Optional[np.ndarray], optional): Disturbance uncertainty kernel . 
            Defaults to Identity matrix.
        soften_state_constraints (bool, optional): Soften state constraints if True 
            Defaults to True.
        softening_penalty (float, optional): State constraint softening penalty.
            Defaults to 1e9.
        input_constraints_flag (bool, optional): Include input constraints if True.
                Defaults to True.
        state_constraints_flag (bool, optional): Include state constraints if True.
                Defaults to True.
    """

    def __init__(self,
                 action_size: int,
                 state_size: int,
                 noise_size: int,
                 output_size: int,
                 horizon: int,
                 rho: float,
                 p_matrix: Optional[np.ndarray] = None,
                 soften_state_constraints: bool = True,
                 softening_penalty: float = 1e9,
                 input_constraints_flag: bool = True,
                 state_constraints_flag: bool = True,
                 ) -> None:
        super().__init__(action_size=action_size,
                         state_size=state_size,
                         noise_size=noise_size,
                         output_size=output_size,
                         horizon=horizon)
        self.soften_state_constraints = soften_state_constraints
        self.softening_penalty = softening_penalty
        self.rho = rho
        self.input_constraints_flag = input_constraints_flag
        self.state_constraints_flag = state_constraints_flag
        if p_matrix is None:
            p_matrix = np.eye(self.noise_size)
        self.p_matrix = p_matrix

    def _full_matrices(self, params: NominalLinearEnvParams) -> Tuple[np.ndarray]:
        """ Compose vectorized matrices

        Returns:
            Tuple[np.ndarray]: Vectorized A, B, E, C, and P matrices of shape:
                (ST, S), (ST, AT), (ST, WT), (ST, ST), and (WT, WT) respectively.
                where S denotes the output/state space size, A denotes the input/action
                space size, T denotes the environment length, and W denotes the state
                disturbance size.
        """
        full_a = np.zeros((self.state_size * self.horizon, self.state_size))
        full_b = np.zeros((self.state_size * self.horizon, self.action_size * self.horizon))
        full_e = np.zeros((self.state_size * self.horizon, self.noise_size * self.horizon))
        full_c = np.kron(np.eye(self.horizon), params.matrices.c_matrix)
        full_p = np.kron(np.eye(self.horizon), self.p_matrix)

        for t in range(self.horizon):
            full_a[self.state_size * t:self.state_size *
                   (t+1), :] = np.linalg.matrix_power(params.matrices.a_matrix, t+1)
            full_b += np.kron(
                np.diag(np.ones(self.horizon-t), k=-t),
                np.linalg.matrix_power(params.matrices.a_matrix, t) @ params.matrices.b_matrix)
            full_e += np.kron(
                np.diag(np.ones(self.horizon-t), k=-t),
                np.linalg.matrix_power(params.matrices.a_matrix, t) @ params.matrices.e_matrix)
        return full_a, full_b, full_e, full_c, full_p

    def prepare_optimizer(self, params: NominalLinearEnvParams) -> Optimizer:
        """ Prepare a parametric optimization problem for the robust mpc agent

        Returns:
            Optimizer: Optimization problem, parameters, and variables
        """
        if not np.allclose(params.matrices.d_matrix, 0, atol=1e-6):
            raise ValueError("D matrix is not supported. It must be all zeros.")
        # raise NotImplementedError("Linearized dynamics are not implemented yet!")
        full_actions = cp.Variable((self.action_size, self.horizon))
        gamma_1 = cp.Variable(1)
        gamma_2 = cp.Variable(1)
        lambda_var = cp.Variable(1)

        x_par = cp.Parameter(self.state_size)
        r_par = cp.Parameter((self.output_size, self.horizon))
        w_par = cp.Parameter((self.noise_size, self.horizon))

        full_a, full_b, full_e, full_c, full_p = self._full_matrices(params)

        full_action_cost = np.kron(np.eye(self.horizon), params.costs.action)
        full_state_cost = np.block([
            [np.kron(np.eye(self.horizon-1), params.costs.state),
             np.zeros((self.output_size * (self.horizon - 1), self.output_size))],
            [np.zeros((self.output_size, self.output_size * (self.horizon - 1))),
             params.costs.final]
        ])

        # First LMI constraint
        lmi_a = full_e.T @ full_c.T @ full_state_cost @ full_c @ full_e - lambda_var * full_p
        lmi_b = cp.reshape(
            full_e.T @ full_c.T @ full_state_cost
            @ (full_c @ (full_a @ x_par + full_b @ full_actions.flatten()) - r_par.flatten())
            + lambda_var * full_p @ w_par.flatten(),
            [self.noise_size * self.horizon, 1])
        lmi_c = cp.reshape(
            (cp.quad_form(r_par.flatten(), full_state_cost)
             - lambda_var * (cp.quad_form(w_par.flatten(), full_p) - self.rho**2) - gamma_1),
            [1, 1])
        lmi_constraint_1 = cp.bmat([
            [lmi_a, lmi_b],
            [lmi_b.T, lmi_c]
        ]) << 0

        # Second LMI constraint
        cost_matrix = full_b.T @ full_c.T @ full_state_cost @ full_c @ full_b + full_action_cost
        lmi_a = -np.eye(cost_matrix.shape[0])
        lmi_b = cp.reshape(
            sqrtm(cost_matrix) @ full_actions.flatten(),
            [cost_matrix.shape[0], 1]
        )
        lmi_c = cp.reshape(
            (cp.quad_form(full_a @ x_par, full_c.T @ full_state_cost @ full_c)
             - 2 * cp.sum(cp.multiply(
                 r_par.flatten(),
                 (full_state_cost @ full_c @ full_a @ x_par)))
             + 2 * cp.sum(cp.multiply(
                 (full_b.T @ full_c.T @ full_state_cost @ (full_c @ full_a @ x_par - r_par.flatten())),
                 full_actions.flatten()))
             - gamma_2),
            [1, 1]
        )
        lmi_constraint_2 = cp.bmat([
            [lmi_a, lmi_b],
            [lmi_b.T, lmi_c]
        ]) << 0

        constraint_lambda = (lambda_var >= 0)
        objective = gamma_1 + gamma_2

        constraints = [constraint_lambda, lmi_constraint_1, lmi_constraint_2]
        if self.input_constraints_flag:
            full_action_constraint_matrix = np.kron(
                np.eye(self.horizon), params.constraints.action_constraint_matrix)
            full_action_constraint_vector = np.kron(
                np.ones((self.horizon)), params.constraints.action_constraint_vector)
            constraints.append(full_action_constraint_matrix @
                               full_actions.flatten() <= full_action_constraint_vector)

        if self.state_constraints_flag:
            full_state_constraint_matrix = np.kron(
                np.eye(self.horizon), params.constraints.state_constraint_matrix)
            full_state_constraint_vector = np.kron(
                np.ones((self.horizon,)), params.constraints.state_constraint_vector)

            # Robustify state constraints
            size = full_state_constraint_matrix.shape[0]
            g_bar_items = []
            product_matrix = full_state_constraint_matrix @ full_e

            for i in range(size):
                gi = product_matrix[i, :].flatten()
                g_bar_items.append(self.rho * cp.sqrt(cp.matrix_frac(gi, sqrtm(full_p))
                                                      ) + cp.sum(cp.multiply(gi, w_par.flatten())))

            g_bar = cp.vstack(g_bar_items).flatten()

            if self.soften_state_constraints:
                slack = cp.Variable(full_state_constraint_vector.shape[0])
                objective += self.softening_penalty * cp.norm(slack, 2)**2
            else:
                slack = np.zeros(full_state_constraint_vector.shape)
            constraints.append(full_state_constraint_matrix @ (full_a @ x_par + full_b @
                                                               full_actions.flatten()) <= full_state_constraint_vector - g_bar + slack)

        prob = cp.Problem(cp.Minimize(objective), constraints)
        return Optimizer(
            problem=prob,
            parameters={"initial_state": x_par,
                        "reference_sequence": r_par,
                        "state_disturbance": w_par},
            variables={"actions": full_actions,
                       "lambda": lambda_var}
        )
