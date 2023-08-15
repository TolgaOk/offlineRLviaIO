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
                 learning_rate: float = 1e-4,
                 lr_exp_decay: float = 0.98,
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

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.train_optimizer, gamma=np.sqrt(lr_exp_decay).item())

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
        if self.state_constraints_flag:
            raise NotImplementedError("Iterative IO Controller does not support state constraints!")

        min_actions = -(torch.linalg.inv(theta_uu) @ theta_su.T @ aug_states.T).T
        if (len(self.params.constraints.action.vector) == 2 * self.action_size
            and np.allclose(self.params.constraints.action.matrix[:self.action_size], np.eye(self.action_size))
                and np.allclose(self.params.constraints.action.matrix[self.action_size:], -np.eye(self.action_size))):
            action_high = torch.from_numpy(self.params.constraints.action.vector[:self.action_size].reshape(
                1, -1)).float().to(aug_states.device)
            action_low = -torch.from_numpy(self.params.constraints.action.vector[self.action_size:].reshape(
                1, -1)).float().to(aug_states.device)
            min_actions = min_actions.clamp(
                min=action_low,
                max=action_high)
        else:
            raise NotImplementedError("Only rectengular action constraints are supported!")

        return min_actions

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

        with tqdm(total=epochs, disable=not verbose) as pbar:
            for epoch in range(epochs):
                dataset_indices = rng.permutation(len(augmented_dataset))
                data_size = len(dataset_indices)

                epoch_losses = []
                for index in range(0, data_size, batch_size):
                    indices = dataset_indices[index: index + batch_size]

                    aug_states = np.stack(
                        [augmented_dataset[_index].aug_state for _index in indices], axis=0)
                    exp_actions = np.stack(
                        [augmented_dataset[_index].expert_action for _index in indices], axis=0)

                    loss = self.loss(
                        torch.from_numpy(aug_states).float(),
                        torch.from_numpy(exp_actions).float()
                    ).mean(-1)

                    self.train_optimizer.zero_grad()
                    loss.backward()
                    self.train_optimizer.step()
                    epoch_losses.append(loss.item())

                train_losses.append(np.mean(epoch_losses))
                self.scheduler.step()
                if verbose:
                    pbar.set_postfix({"loss": train_losses[-1], "lr": self.scheduler.get_lr()[-1]})
                    pbar.update()

        self._q_theta_uu = self.th_theta_uu.weight.cpu().detach().numpy()
        self._q_theta_su = self.th_theta_su.weight.cpu().detach().numpy()
        return train_losses
