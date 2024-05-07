from typing import Callable, List, Tuple, Optional, Any, Union, Dict, NamedTuple
from functools import partial
from tqdm.notebook import tqdm
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrd
import jaxopt
import optax
from jaxtyping import Array, Float, ArrayLike

from io_agent.plant.base import (NominalLinearEnvParams,
                                 LinearConstraints)
from io_agent.utils import AugmentedTransition, FeatureHandler
from io_agent.control.io import IOController
from io_agent.plant.base import NominalLinearEnvParams, LinearConstraints


class IOParams(NamedTuple):
    theta_uu: Float[Array, "A A"]
    theta_su: Float[Array, "S A"]


class JaxIOController(IOController):

    def __init__(self,
                 constraints: LinearConstraints,
                 feature_handler: FeatureHandler,
                 key: ArrayLike,
                 include_constraints: bool = True,
                 action_constraints_flag: bool = True,
                 state_constraints_flag: bool = True,
                 learning_rate: float = 1e-4,
                 lr_exp_decay: float = 0.975,
                 scheduler_transition_step: int = 5000,
                 ):
        self.learning_rate = learning_rate
        self.lr_exp_decay = lr_exp_decay
        self.key = key
        self.scheduler = optax.exponential_decay(
            learning_rate,
            transition_steps=scheduler_transition_step,
            decay_rate=lr_exp_decay)
        self._last_lr = learning_rate
        super().__init__(
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

        self.const_matrix = self.params.constraints.action.matrix
        self.const_vector = self.params.constraints.action.vector

        if self.state_constraints_flag:
            raise NotImplementedError(
                "Iterative IO Controller does not support state constraints!")

        if not (np.allclose(self.const_matrix[:self.action_size], np.eye(self.action_size)) and
                np.allclose(self.const_matrix[self.action_size:], -np.eye(self.action_size)) and
                np.allclose(self.const_vector, np.ones(self.action_size * 2))):
            raise NotImplementedError(
                "Only unit rectangular action constraints are supported!")

    def state_dict(self) -> Dict[str, Any]:
        return dict(
            thetas=self.theta_param,
            constraints=self.params.constraints,
            feature_handler=self.feature_handler,
            include_constraints=self.include_constraints,
            action_constraints_flag=self.action_constraints_flag,
            state_constraints_flag=self.state_constraints_flag,
            learning_rate=self.learning_rate,
            lr_exp_decay=self.lr_exp_decay,
        )

    @staticmethod
    def load_states(states: Dict[str, Any]) -> "JaxIOController":
        raise NotImplementedError

    def prepare_train_optimizer(self) -> None:
        self.key, init_key = jrd.split(self.key, 2)
        self.theta_param = init_params(
            init_key, self.action_size, self.aug_state_size)
        return jaxopt.GradientDescent(
            # fun=jax.jit(batch_loss_fn),
            fun=batch_loss_fn,
            # max_stepsize=1.0
            # opt=optax.adam(learning_rate=self.learning_rate)
            stepsize=self.scheduler,
            acceleration=False
        )

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
        aug_state = jnp.asarray(
            self.feature_handler.augment_state(state, self._history))
        min_action = minimizer_action(self.theta_param, aug_state)
        self._past_state = state.copy()
        self._past_action = min_action.copy()
        return min_action, None

    def train(self,
              augmented_dataset: List[AugmentedTransition],
              batch_size: int,
              ) -> None:
        states = jnp.stack(
            [tran.aug_state for tran in augmented_dataset])
        exp_actions = jnp.stack(
            [tran.expert_action for tran in augmented_dataset])

        opt_state = self.train_optimizer.init_state(
            jax.tree_util.tree_map(jnp.zeros_like, self.theta_param),
            jnp.zeros_like(states[:1]),
            jnp.zeros_like(exp_actions[:1]))
        step_count = 0

        while True:
            self.key, permute_key = jrd.split(self.key, 2)
            dataset_indices = jrd.permutation(
                permute_key, len(augmented_dataset))
            data_size = len(dataset_indices)

            within_epoch_losses = []
            for index in range(0, data_size, batch_size):
                indices = dataset_indices[index: index + batch_size]
                param, opt_state = self.train_optimizer.update(
                    self.theta_param,
                    opt_state,
                    states[indices],
                    exp_actions[indices])
                self.theta_param = project_theta_uu(param)
                loss = batch_loss_fn(
                    param, states[indices], exp_actions[indices])
                self._last_lr = opt_state.stepsize.item()
                opt_error = opt_state.error
                within_epoch_losses.append(loss.item())
                yield np.mean(within_epoch_losses).item()

            self._q_theta_uu = self.theta_param.theta_uu
            self._q_theta_su = self.theta_param.theta_su


def init_params(key: ArrayLike, action_size: int, state_size: int) -> IOParams:
    eig_val_key, eig_vec_key, su_key = jrd.split(key, 3)

    eig_vecs = jrd.orthogonal(eig_vec_key, action_size)
    eig_vals = jax.nn.softplus(jrd.normal(
        eig_val_key, (action_size,))) + 1
    init_psd_matrix = eig_vecs @ jnp.diag(eig_vals) @ eig_vecs.T

    return IOParams(
        theta_uu=init_psd_matrix,
        theta_su=jrd.normal(
            su_key, (state_size, action_size))
    )


def project_theta_uu(param: IOParams) -> IOParams:
    eig_vals, eig_vecs = jnp.linalg.eigh(param.theta_uu)
    eig_vals = jnp.clip(eig_vals, a_min=1.0 + 1e-4)
    return IOParams(
        theta_uu=eig_vecs @ jnp.diag(eig_vals) @ eig_vecs.T,
        theta_su=param.theta_su
    )


def q_fn(param: IOParams,
         state: Float[Array, "S"],
         action: Float[Array, "A"],
         ) -> Float[Array, ""]:
    return 2 * state @ param.theta_su @ action + action @ param.theta_uu @ action


@partial(jax.vmap, in_axes=(None, 0, 0))
def loss_fn(param: IOParams,
            state: Float[Array, "S"],
            exp_action: Float[Array, "A"],
            ) -> Float[Array, ""]:
    min_action = jax.lax.stop_gradient(minimizer_action(param, state))
    return q_fn(param, state, exp_action) - q_fn(param, state, min_action)


@jax.jit
def batch_loss_fn(param: IOParams,
                  state: Float[Array, "B S"],
                  exp_action: Float[Array, "B A"],
                  ) -> Float[Array, ""]:
    return loss_fn(param, state, exp_action).mean(axis=-1)


@jax.jit
def minimizer_action(param: IOParams,
                     state: Float[Array, "S"],
                     box_low: float = -1.0,
                     box_high: float = 1.0,
                     ) -> Float[Array, "A"]:

    qp = jaxopt.BoxCDQP(jit=True)
    action_size = param.theta_uu.shape[0]
    init_action = jnp.zeros(action_size)

    Q_matrix = param.theta_uu
    c_vector = state @ param.theta_su

    sol = qp.run(init_action,
                 params_obj=(Q_matrix, c_vector),
                 params_ineq=(jnp.ones(action_size) * box_low,
                              jnp.ones(action_size) * box_high)
                 ).params

    return sol
