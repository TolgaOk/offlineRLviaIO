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
                 device: str = "cpu",
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
        self.device = device
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.train_optimizer, gamma=np.sqrt(lr_exp_decay).item())
        self.to(device)

        self.const_matrix = self.params.constraints.action.matrix
        self.const_vector = self.params.constraints.action.vector

        if self.state_constraints_flag:
            raise NotImplementedError("Iterative IO Controller does not support state constraints!")

        if (len(self.const_vector) == 2 * self.action_size
            and np.allclose(self.const_matrix[:self.action_size], np.eye(self.action_size))
                and np.allclose(self.const_matrix[self.action_size:], -np.eye(self.action_size))):
            self.action_high = self.to_torch(self.const_vector[:self.action_size].reshape(1, -1))
            self.action_low = -self.to_torch(self.const_vector[self.action_size:].reshape(1, -1))
        else:
            raise NotImplementedError("Only rectengular action constraints are supported!")

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

        # self.theta_su_net = torch.nn.Sequential(
        #     torch.nn.Linear(self.aug_state_size, 128),
        #     torch.nn.Elu(),
        #     torch.nn.Linear(128, 128),
        #     torch.nn.Elu(),
        #     torch.nn.Linear(128, self.action_size * self.aug_state_size),
        # )
        # self.theta_uu_net = torch.nn.Sequential(
        #     torch.nn.Linear(self.aug_state_size, 128),
        #     torch.nn.Elu(),
        #     torch.nn.Linear(128, 128),
        #     torch.nn.Elu(),
        #     torch.nn.Linear(128, (self.action_size ** 2)),
        #     torch.nn.Unflatten(-1, (self.action_size, self.action_size)),
        #     geotorch.PSD((self.action_size, self.action_size),
        #                  f=(self.softplus_epsilon, inv_softplus_epsilon),
        #                  triv="expm")
        # )
        self.th_theta_su = torch.nn.Linear(self.action_size, self.aug_state_size, bias=False)
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def q_fn(self,
             state: torch.Tensor,
             action: torch.Tensor,
             theta_uu: torch.Tensor,
             theta_su: torch.Tensor,
             ) -> torch.Tensor:
        return (2 * (state * (theta_su @ action.T).T).sum(dim=-1)
                + (action * (theta_uu @ action.T).T).sum(dim=-1))

    def batch_minimizer_actions(self,
                                aug_states: torch.Tensor,
                                theta_uu: torch.Tensor,
                                theta_su: torch.Tensor,
                                ) -> torch.Tensor:
        min_actions = -(torch.linalg.inv(theta_uu) @ theta_su.T @ aug_states.T).T
        min_actions = min_actions.clamp(
            min=self.action_low,
            max=self.action_high)
        return min_actions

    def compute(self,
                state: np.ndarray,
                reference: np.ndarray,
                ) -> Tuple[Union[np.ndarray, float]]:
        if self._past_state is not None and self._past_action is not None:
            self._history = self.feature_handler.update_history(
                state=self._past_state,
                action=self._past_action,
                next_state=state,
                history=self._history)
        th_aug_state = self.to_torch(self.feature_handler.augment_state(
            state, self._history).reshape(1, -1))
        kwargs = dict(aug_states=th_aug_state,
                      theta_su=self.th_theta_su.weight,
                      theta_uu=self.th_theta_uu.weight)
        th_action = self.batch_minimizer_actions(**kwargs)
        action = th_action.detach().cpu().numpy().reshape(-1)
        self._past_state = state.copy()
        self._past_action = action.copy()
        return action, None

    def loss(self, aug_states: torch.Tensor, exp_action: torch.Tensor) -> torch.Tensor:
        theta_uu = self.th_theta_uu.weight
        min_actions = self.batch_minimizer_actions(
            aug_states, theta_uu=theta_uu, theta_su=self.th_theta_su.weight)
        return (self.q_fn(aug_states, exp_action, theta_uu=theta_uu, theta_su=self.th_theta_su.weight)
                - self.q_fn(aug_states, min_actions, theta_uu=theta_uu, theta_su=self.th_theta_su.weight))

    def to_torch(self, array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array).float().to(self.device)

    def train(self,
              augmented_dataset: List[AugmentedTransition],
              epochs: int,
              batch_size: int,
              rng: np.random.Generator,
              verbose: bool = True,
              ) -> None:
        train_losses = []

        th_aug_states = self.to_torch(np.stack([tran.aug_state for tran in augmented_dataset]))
        th_exp_actions = self.to_torch(np.stack([tran.expert_action for tran in augmented_dataset]))

        with tqdm(total=epochs, disable=not verbose) as pbar:
            for epoch in range(epochs):
                dataset_indices = rng.permutation(len(augmented_dataset))
                data_size = len(dataset_indices)

                epoch_losses = []
                for index in range(0, data_size, batch_size):
                    indices = dataset_indices[index: index + batch_size]

                    loss = self.loss(
                        th_aug_states[indices],
                        th_exp_actions[indices]
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
