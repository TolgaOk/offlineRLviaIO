from typing import Callable, List, Tuple, Optional, Any, Union, Dict
from functools import partial
from tqdm.notebook import tqdm
import numpy as np
from qpth.qp import QPFunction
import torch
import geotorch

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
        self.lr_exp_decay = lr_exp_decay
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
            raise NotImplementedError("Only rectangular action constraints are supported!")
        self.to(device)

    def get_states(self) -> Dict[str, Any]:
        return dict(
            parameters={key: param.cpu() for key, param in self.state_dict().items()},
            constraints=self.params.constraints,
            feature_handler=self.feature_handler,
            include_constraints=self.include_constraints,
            action_constraints_flag=self.action_constraints_flag,
            state_constraints_flag=self.state_constraints_flag,
            learning_rate=self.learning_rate,
            lr_exp_decay=self.lr_exp_decay,
            device=self.device,
        )

    @staticmethod
    def load_states(states: Dict[str, Any]) -> "IterativeIOController":
        parameters = states.pop("parameters")
        controller = IterativeIOController(**states)
        controller.load_state_dict(parameters)
        return controller

    def to_torch(self, array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array).float().to(self.device)

    def prepare_train_optimizer(self) -> torch.optim.Optimizer:
        so_module = geotorch.SO((self.action_size, self.action_size))
        eig_vals = torch.nn.functional.softplus(torch.randn(self.action_size)) + 1
        eig_vecs = so_module.sample()
        init_psd_matrix = eig_vecs @ torch.diag(eig_vals) @ eig_vecs.T

        self.th_theta_uu = torch.nn.Parameter(init_psd_matrix)
        self.th_theta_su = torch.nn.Parameter(torch.randn(self.aug_state_size, self.action_size))
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate)

    def project_theta_uu(self):
        eig_vals, eig_vecs = torch.linalg.eigh(self.th_theta_uu.data)
        eig_vals = eig_vals.clamp(min=1.0 + 1e-4)
        self.th_theta_uu.data = eig_vecs @ torch.diag(eig_vals) @ eig_vecs.T

    def q_fn(self,
             state: torch.Tensor,
             action: torch.Tensor,
             ) -> torch.Tensor:
        return (2 * (state * (self.th_theta_su @ action.T).T).sum(dim=-1)
                + (action * (self.th_theta_uu @ action.T).T).sum(dim=-1))

    def batch_minimizer_actions(self,
                                aug_states: torch.Tensor,
                                ) -> torch.Tensor:
        if self.state_constraints_flag:
            raise NotImplementedError("Iterative IO Controller does not support state constraints!")
        eig_val, _ = torch.linalg.eigh(self.th_theta_uu.data)
        assert torch.all(eig_val >= 1.0), eig_val
        assert torch.allclose(self.th_theta_uu.data, self.th_theta_uu.data.T, atol=1e-6)

        return QPFunction(verbose=-1)(
            self.th_theta_uu,
            aug_states @ self.th_theta_su,
            self.to_torch(self.const_matrix),
            self.to_torch(self.const_vector),
            torch.Tensor().to(self.device),
            torch.Tensor().to(self.device))

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
        th_action = self.batch_minimizer_actions(aug_states=th_aug_state)

        action = th_action.detach().cpu().numpy().reshape(-1)
        self._past_state = state.copy()
        self._past_action = action.copy()
        return action, None

    def loss(self, aug_states: torch.Tensor, exp_action: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            min_actions = self.batch_minimizer_actions(aug_states)
        return (self.q_fn(aug_states, exp_action) - self.q_fn(aug_states, min_actions))

    def train(self,
              augmented_dataset: List[AugmentedTransition],
              batch_size: int,
              rng: np.random.Generator,
              ) -> None:
        th_aug_states = self.to_torch(np.stack([tran.aug_state for tran in augmented_dataset]))
        th_exp_actions = self.to_torch(np.stack([tran.expert_action for tran in augmented_dataset]))

        while True:
            dataset_indices = rng.permutation(len(augmented_dataset))
            data_size = len(dataset_indices)

            within_epoch_losses = []
            for index in range(0, data_size, batch_size):
                indices = dataset_indices[index: index + batch_size]

                loss = self.loss(
                    th_aug_states[indices],
                    th_exp_actions[indices]
                ).mean(-1)

                self.train_optimizer.zero_grad()
                loss.backward()
                self.train_optimizer.step()
                self.project_theta_uu()
                within_epoch_losses.append(loss.item())
                yield loss.item()

            self.scheduler.step()
            self._q_theta_uu = self.th_theta_uu.cpu().detach().numpy()
            self._q_theta_su = self.th_theta_su.cpu().detach().numpy()
            # yield np.mean(within_epoch_losses), within_epoch_losses
