from typing import Callable, List, Tuple, Optional, Any, Union, Dict
from functools import partial
from tqdm.notebook import tqdm
import numpy as np
import torch
import geotorch
from geotorch.fixedrank import softplus_epsilon, inv_softplus_epsilon

from io_agent.plant.base import (NominalLinearEnvParams,
                                 LinearConstraints)
from io_agent.utils import AugmentedTransition, FeatureHandler

from io_agent.control.io import IOController
from io_agent.plant.base import NominalLinearEnvParams, LinearConstraints


class IterativeIOController(torch.nn.Module, IOController):

    def __init__(self,
                 constraints: LinearConstraints,
                 feature_handler: FeatureHandler,
                 include_constraints: bool = True,
                 action_constraints_flag: bool = True,
                 state_constraints_flag: bool = True,
                 learning_rate: float = 1e-4
                 ):
        super().__init__()
        self.learning_rate = learning_rate
        IOController.__init__(
            self,
            params=NominalLinearEnvParams(
                matrices=None,
                constraints=constraints,
                costs=None),
            dataset_length=None,
            feature_handler=feature_handler,
            include_constraints=include_constraints,
            action_constraints_flag=action_constraints_flag,
            state_constraints_flag=state_constraints_flag,
            soften_state_constraints=True,
            softening_penalty=1e9,
        )

    @staticmethod
    def softplus_epsilon(x, epsilon=1e-6):
        return torch.nn.functional.softplus(x) + epsilon + 1

    def prepare_train_optimizer(self) -> torch.optim.Optimizer:
        self.th_theta_uu = torch.nn.Linear(self.action_size, self.action_size, bias=False)
        geotorch.positive_definite(
            module=self.th_theta_uu,
            tensor_name="weight",
            f=(self.softplus_epsilon, inv_softplus_epsilon),
            triv="expm"
        )
        self.th_theta_su = torch.nn.Linear(self.action_size, self.aug_state_size, bias=False)

        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def q_fn(self,
             state: torch.Tensor,
             action: torch.Tensor,
             theta_uu: Optional[torch.Tensor]
             ) -> torch.Tensor:
        theta_uu = theta_uu if theta_uu is not None else self.th_theta_uu.weight
        return (2 * (state * (self.th_theta_su.weight @ action.T).T).sum(dim=-1)
                + (action * (theta_uu @ action.T).T).sum(dim=-1))

    def batch_minimizer_actions(self,
                                aug_states: torch.Tensor,
                                theta_uu: torch.Tensor,
                                theta_su: torch.Tensor,
                                ) -> torch.Tensor:
        batch_size, _ = aug_states.shape
        actions = np.zeros((batch_size, self.action_size))
        self._q_theta_uu = theta_uu.cpu().detach().numpy()
        self._q_theta_su = theta_su.cpu().detach().numpy()
        action_optimizer = self.prepare_action_optimizer()

        for index, aug_state in enumerate(aug_states.cpu().detach().numpy()):
            if self.include_constraints:
                state = self.feature_handler.original_state(aug_state)
                constraint_matrix, constraint_vector = self.calculate_constraints(state)
                action_optimizer.parameters["constraint_matrix"].value = constraint_matrix
                action_optimizer.parameters["constraint_vector"].value = constraint_vector
            action_optimizer.parameters["state"].value = aug_state
            action_optimizer.problem.solve()
            if action_optimizer.problem.status in ("infeasible", "unbounded"):
                raise RuntimeError(
                    f"Action optimization failed with status: {action_optimizer.problem.status}")
            actions[index] = action_optimizer.variables["action"].value

        return torch.from_numpy(actions).float().to(aug_states.device)

    def loss(self, aug_states: torch.Tensor, exp_action: torch.Tensor) -> torch.Tensor:
        theta_uu = self.th_theta_uu.weight
        min_actions = self.batch_minimizer_actions(aug_states, theta_uu, self.th_theta_su.weight)
        return (self.q_fn(aug_states, exp_action, theta_uu=theta_uu)
                - self.q_fn(aug_states, min_actions, theta_uu=theta_uu))

    def train(self,
              augmented_dataset: List[AugmentedTransition],
              epochs: int,
              batch_size: int,
              rng: np.random.Generator,
              verbose: bool = True
              ) -> None:
        train_losses = []

        loading_bar = (partial(tqdm)
                       if verbose else lambda x: x)
        for epoch in loading_bar(range(epochs)):
            dataset_indices = rng.permutation(len(augmented_dataset))
            data_size = len(dataset_indices)

            epoch_losses = []
            for index in range(0, data_size, batch_size):
                indices = dataset_indices[index: index + batch_size]

                aug_states = np.stack(
                    [augmented_dataset[_index].aug_state for _index in indices], axis=0)
                exp_actions = np.stack(
                    [augmented_dataset[_index].action for _index in indices], axis=0)

                loss = self.loss(
                    torch.from_numpy(aug_states).float(),
                    torch.from_numpy(exp_actions).float()
                ).mean(-1)

                self.train_optimizer.zero_grad()
                loss.backward()
                self.train_optimizer.step()
                epoch_losses.append(loss.item())

            train_losses.append(np.mean(epoch_losses))
            if verbose:
                print(f"Epoch: {epoch + 1}, loss: {train_losses[-1]}")

        self._q_theta_uu = self.th_theta_uu.weight.cpu().detach().numpy()
        self._q_theta_su = self.th_theta_su.weight.cpu().detach().numpy()
        return train_losses
