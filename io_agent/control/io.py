from typing import Any, Tuple, Dict, Optional, Union, List
import numpy as np
from scipy.linalg import sqrtm
import cvxpy as cp
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, asdict
from collections import deque

from io_agent.plant.base import NominalLinearEnvParams
from io_agent.control.mpc import Optimizer
from io_agent.utils import FeatureHandler, AugmentedTransition
from io_agent.evaluator import Transition
from io_agent.control.mpc import MPC
from io_agent.control.rmpc import RobustMPC


class IOController():
    """ Inverse Optimization based controller agent

    Args:
        params (NominalLinearEnvParams): Linear environment parameters
        dataset_length (int): MPC dataset_length
    """

    def __init__(self,
                 params: NominalLinearEnvParams,
                 dataset_length: int,
                 feature_handler: FeatureHandler,
                 include_constraints: bool = True,
                 action_constraints_flag: bool = True,
                 state_constraints_flag: bool = True,
                 soften_state_constraints: bool = True,
                 softening_penalty: float = 1e9,
                 ) -> None:
        self.feature_handler = feature_handler
        self.include_constraints = include_constraints and (params.constraints.state or params.constraints.action)
        self.action_constraints_flag = action_constraints_flag
        self.state_constraints_flag = state_constraints_flag
        self.soften_state_constraints = soften_state_constraints
        self.softening_penalty = softening_penalty
        self.dataset_length = dataset_length
        self.params = params
        self.horizon = None  # For compatibility with the ControlLoop class

        action_const_size = (0 if params.constraints.action is None else params.constraints.action.vector.shape[0])
        state_const_size = (0 if params.constraints.state is None else params.constraints.state.vector.shape[0])
        self.polytope_size = (
            action_const_size * int(action_constraints_flag)
            + state_const_size * int(state_constraints_flag)
        )
        self.action_size = self.feature_handler.action_size
        self.state_size = self.feature_handler.state_size
        self.aug_state_size = self.feature_handler.aug_state_size
        self.train_optimizer = self.prepare_train_optimizer()

        self._past_state = None
        self._past_action = None
        self._q_theta_su = None
        self._q_theta_uu = None
        self._history = None

    def compute(self,
                state: np.ndarray,
                reference: np.ndarray,
                ) -> Tuple[Union[np.ndarray, float]]:
        """ Compute the optimal action and total cost given the state

        Args:
            state (np.ndarray): State array of shape (S,)
                where S denotes the output/state space size
            reference (np.ndarray): Reference signal array to follow of shape (S, T)
                where L denotes the environment length

        Raises:
            RuntimeError: If the optimization is failed

        Returns:
            Tuple[Union[np.ndarray, float]]:
                - Optimal action vector (if possible) of shape (A,) at the initial time
                    step where A denotes the action space size or None if the solver
                    cannot solve the problem.
                - Total cost of the MPC problem (float)
        """
        if self._q_theta_su is None or self._q_theta_uu is None:
            return np.zeros((self.action_size,))
        if self._past_state is not None and self._past_action is not None:
            self._history = self.feature_handler.update_history(
                state=self._past_state,
                action=self._past_action,
                next_state=state,
                history=self._history)
        aug_state = self.feature_handler.augment_state(state, self._history)
        self.action_optimizer.parameters["state"].value = aug_state
        if self.include_constraints:
            constraint_matrix, constraint_vector = self.calculate_constraints(state)
            self.action_optimizer.parameters["constraint_matrix"].value = constraint_matrix
            self.action_optimizer.parameters["constraint_vector"].value = constraint_vector
        self.action_optimizer.problem.solve()
        if self.action_optimizer.problem.status in ("infeasible", "unbounded"):
            raise RuntimeError(
                f"Action optimization failed with status: {self.action_optimizer.problem.status}")
        self._past_state = state.copy()
        self._past_action = self.action_optimizer.variables["action"].value.copy()

        return self.action_optimizer.variables["action"].value, None

    def reset(self) -> None:
        """ Reset the feature handler
        """
        self._past_state = None
        self._past_action = None
        self._history = self.feature_handler.reset_history()

    def prepare_action_optimizer(self) -> Optimizer:
        """ Prepare the action generation optimizer that is used in ```compute```

        Returns:
            Optimizer: Optimization problem, parameters, and variables
        """
        state = cp.Parameter((self.aug_state_size,))
        action = cp.Variable((self.action_size,))

        if self.include_constraints:
            constraint_matrix = cp.Parameter((self.polytope_size, self.action_size))
            constraint_vector = cp.Parameter((self.polytope_size,))

        objective = (
            cp.quad_form(action, (self._q_theta_uu + self._q_theta_uu.T) / 2)
            + 2 * cp.sum(cp.multiply(state, self._q_theta_su @ action))
        )

        if (self.state_constraints_flag and self.params.constraints.state) and self.soften_state_constraints:
            slack_state = cp.Variable(self.params.constraints.state.vector.shape[0])
            objective += + self.softening_penalty * cp.norm(slack_state, 2)**2
            slack_action = cp.Variable(
                self.polytope_size - self.params.constraints.state.vector.shape[0])
            slack = cp.hstack([slack_action, slack_state])

        constraints = []
        if self.include_constraints:
            constraints.append(constraint_matrix @ action <= constraint_vector + slack)
        objective = cp.Minimize(objective)
        problem = cp.Problem(objective, constraints)

        parameters = {
            "state": state,
        }
        if self.include_constraints:
            parameters["constraint_matrix"] = constraint_matrix
            parameters["constraint_vector"] = constraint_vector

        return Optimizer(
            problem=problem,
            parameters=parameters,
            variables={
                "action": action,
            }
        )

    def calculate_constraints(self, state: np.ndarray) -> Tuple[np.ndarray]:
        """ Calculate the constraint matrices and vectors for the io agent training 

        Args:
            state (np.ndarray): State array of shape (S,)
                where S denotes the output/state space size

        Returns:
            Tuple[np.ndarray, float]:
                - constraint_matrix (np.ndarray)
                - constraint_vector (np.ndarray)
        """
        constraint_matrices = []
        constraint_vectors = []
        if (self.state_constraints_flag) and (self.params.constraints.state is not None):
            constraint_matrices.append(
                self.params.constraints.state.matrix @ self.params.matrices.b_matrix
            )
            constraint_vectors.append(
                self.params.constraints.state.vector
                - self.params.constraints.state.matrix @ self.params.matrices.a_matrix @ state
            )
        if (self.action_constraints_flag) and (self.params.constraints.action is not None):
            constraint_matrices.append(self.params.constraints.action.matrix)
            constraint_vectors.append(self.params.constraints.action.vector)
        constraint_matrix = np.concatenate(constraint_matrices, axis=0)
        constraint_vector = np.concatenate(constraint_vectors, axis=0)
        return (constraint_matrix, constraint_vector)

    def train(self, augmented_dataset: List[AugmentedTransition], rng: np.random.Generator) -> None:
        """ Train the io agent with the given dataset of augmented transitions

        Args:
            dataset (List[AugmentedTransition]): List of  transitions that includes augmented
                states

        Raises:
            RuntimeError: If the optimization is failed
        """
        if len(augmented_dataset) < self.dataset_length:
            raise ValueError(
                f"Given dataset length: {self.dataset_length} is greater than the available data: {len(augmented_dataset)}")
        dataset_indices = rng.permutation(len(augmented_dataset))[:self.dataset_length]
        transitions = [augmented_dataset[index] for index in dataset_indices]

        states = np.stack([transition.aug_state for transition in transitions], axis=-1)
        actions = np.stack([transition.expert_action for transition in transitions], axis=-1)
        self.train_optimizer.parameters["states"].value = states
        self.train_optimizer.parameters["actions"].value = actions

        if self.include_constraints:
            for transition in transitions:
                constraint_matrix, constraint_vector = self.calculate_constraints(transition.state)
                transition.constraint_matrix = constraint_matrix
                transition.constraint_vector = constraint_vector

            constraint_matrices = [transition.constraint_matrix for transition in transitions]
            constraint_vectors = np.stack(
                [transition.constraint_vector for transition in transitions], axis=1)
            for index, value in enumerate(constraint_matrices):
                self.train_optimizer.parameters["constraint_matrices"][index].value = value
            self.train_optimizer.parameters["constraint_vector"].value = constraint_vectors

        self.train_optimizer.problem.solve(solver=cp.MOSEK)

        if self.train_optimizer.problem.status in ("infeasible", "unbounded"):
            raise RuntimeError(
                f"Train optimization failed with status: {self.train_optimizer.problem.status}")

        self._q_theta_su = self.train_optimizer.variables["q_theta_su"].value
        self._q_theta_uu = self.train_optimizer.variables["q_theta_uu"].value

    def prepare_train_optimizer(self) -> Optimizer:
        """ Prepare the trainer optimizer

        Returns:
            Optimizer: Optimization problem, parameters, and variables
        """
        states = cp.Parameter((self.aug_state_size, self.dataset_length))
        actions = cp.Parameter((self.action_size, self.dataset_length))
        gamma_var = cp.Variable((1, self.dataset_length))

        if self.include_constraints:
            constraint_matrices = [cp.Parameter(
                (self.polytope_size, self.action_size)) for _ in range(self.dataset_length)]
            constraint_vector = cp.Parameter((self.polytope_size, self.dataset_length))
            lambda_var = cp.Variable((self.polytope_size, self.dataset_length))

        q_theta_su = cp.Variable((self.aug_state_size, self.action_size))
        q_theta_uu = cp.Variable((self.action_size, self.action_size))

        objective = (cp.sum(gamma_var) / 4
                     + cp.sum(cp.multiply(actions, q_theta_uu @ actions))
                     + 2 * cp.sum(cp.multiply(states, q_theta_su @ actions)))
        constraints = [q_theta_uu >> np.eye(self.action_size)]

        if self.include_constraints:
            objective += cp.sum(cp.multiply(lambda_var, constraint_vector))
            constraints.append(lambda_var >= 0)

        q_theta_su_mul_state = q_theta_su.T @ states
        for step in range(self.dataset_length):

            if self.include_constraints:
                lmi_b = cp.reshape(
                    constraint_matrices[step].T @ lambda_var[:, step]
                    + 2 * q_theta_su_mul_state[:, step],
                    [self.action_size, 1], order="C")
                constraints.append(cp.bmat([
                    [q_theta_uu,  lmi_b],
                    [lmi_b.T,  cp.reshape(gamma_var[:, step], [1, 1], order="C")]
                ]) >> 0)
            else:
                lmi_b = cp.reshape(2 * q_theta_su_mul_state[:, step],
                                   [self.action_size, 1], order="C")
                constraints.append(cp.bmat([
                    [q_theta_uu,  lmi_b],
                    [lmi_b.T,  cp.reshape(gamma_var[:, step], [1, 1], order="C")]
                ]) >> 0)

        objective = cp.Minimize(objective)
        problem = cp.Problem(objective, constraints)

        parameters = {
            "states": states,
            "actions": actions,
        }
        if self.include_constraints:
            parameters["constraint_matrices"] = constraint_matrices
            parameters["constraint_vector"] = constraint_vector

        return Optimizer(
            problem=problem,
            parameters=parameters,
            variables={
                "q_theta_su": q_theta_su,
                "q_theta_uu": q_theta_uu,
            }
        )


class AugmentDataset():
    """ Update the given trajectories with the actions of the expert agent and augment
    states with the feature handler
        "Algorithm 1 Using in-hindsight information for IO"

    Args:
        expert_agent (Union[MPC, RobustMPC]): Expert MPC agent
        feature_handler (FeatureHandler): State augmenter
    """

    def __init__(self, expert_agent: Union[MPC, RobustMPC], feature_handler: FeatureHandler) -> None:
        self.expert_agent = expert_agent
        self.feature_handler = feature_handler

    def _get_expert_action(self, state: np.ndarray, noise_sequence: np.ndarray) -> np.ndarray:
        """ Compute the expert action using the expert agent

        Args:
            state (np.ndarray): State array of shape (S,)
                where S denotes the output/state space size
            noise_sequence (np.ndarray): State disturbance array of
                shape (W, T). Where W denotes the length of the state
                disturbance dimension.

        Returns:
            np.ndarray: expert action
        """
        action, _ = self.expert_agent.compute(
            initial_state=state,
            reference_sequence=np.zeros((self.expert_agent.output_size, self.expert_agent.horizon)),
            output_disturbance=np.zeros((self.expert_agent.output_size, self.expert_agent.horizon)),
            state_disturbance=noise_sequence
        )
        return action

    def __call__(self, trajectories: List[List[Transition]]) -> List[AugmentedTransition]:
        """ Prepare the given trajectories by augmenting the states and calculating
            the expert action.

        Args:
            trajectories (List[List[Transition]]): List of trajectories that contains transitions

        Returns:
            List[AugmentedTransition]: Flatten list of augmented transitions
        """
        all_transitions = []
        for traj_index, trajectory in enumerate(trajectories):
            history = self.feature_handler.reset_history()
            noise_queue = deque(maxlen=self.expert_agent.horizon)
            for tran_index, transition in enumerate(trajectory):
                if tran_index >= self.expert_agent.horizon + self.feature_handler.n_past:
                    hindsighted_tran = trajectory[tran_index - self.expert_agent.horizon]
                    expert_action = self._get_expert_action(
                        hindsighted_tran.state,
                        noise_sequence=np.stack(noise_queue, axis=1)
                    )
                    all_transitions.append(
                        AugmentedTransition(
                            aug_state=self.feature_handler.augment_state(
                                state=hindsighted_tran.state, history=history),
                            expert_action=expert_action,
                            **asdict(hindsighted_tran)
                        )
                    )
                noise_queue.append(
                    self.feature_handler.infer_noise(
                        state=transition.state,
                        next_state=transition.next_state,
                        action=transition.action,
                    )
                )
                if tran_index >= self.expert_agent.horizon:
                    hindsighted_tran = trajectory[tran_index - self.expert_agent.horizon]
                    history = self.feature_handler.update_history(
                        state=hindsighted_tran.state,
                        next_state=hindsighted_tran.next_state,
                        action=hindsighted_tran.action,
                        history=history)
        return all_transitions
