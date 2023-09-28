from typing import Any, Tuple, Dict, Optional, Union
import numpy as np
import gym
import d4rl
import jax.numpy as jnp
import jax
import jaxopt

from io_agent.plant.base import (NominalLinearEnvParams,
                                 LinearDiscreteSystem,
                                 LinearConstraints)
from io_agent.plant.base import (Plant,
                                 NominalLinearEnvParams,
                                 LinearConstraints,
                                 LinearConstraint)
from io_agent.utils import FeatureHandler


class MuJoCoEnv(Plant):

    def __init__(self, env: gym.wrappers.TimeLimit) -> None:
        self.env = env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        super().__init__(
            state_size=self.observation_space.shape[0],
            action_size=self.action_space.shape[0],
            noise_size=0,
            output_size=self.observation_space.shape[0],
            max_length=self.env._max_episode_steps,
            costs=None,
            constraints=LinearConstraints(
                action=LinearConstraint(
                    matrix=np.block([
                        [np.eye(self.env.action_space.shape[0])],
                        [-np.eye(self.env.action_space.shape[0])]]),
                    vector=np.concatenate([
                        self.env.action_space.high,
                        -self.env.action_space.low
                    ])
                )
            ),
            reference_sequence=None,
            disturbance_bias=None,
        )

    def nominal_model(self, *args, **kwargs) -> Any:
        return NominalLinearEnvParams(
            matrices=None,
            constraints=self.constraints,
            costs=None
        )

    def symbolic_dynamical_system(self) -> None:
        raise NotImplementedError

    def fill_symbols(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def generate_disturbance(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def reset(self,
              seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None
              ) -> Tuple[Union[np.ndarray, Optional[Dict[str, Any]]]]:
        self._state = self.env.reset(seed=seed)
        return self._state, {"disturbance": None}

    def default_lin_point(self) -> None:
        raise NotImplementedError

    def step(self,
             action: np.ndarray
             ) -> Tuple[Union[np.ndarray, float, bool, Optional[Dict[str, Any]]]]:
        self._state, reward, done, info = self.env.step(action)
        return self._state, reward, done, False, info

    def render(self, *args, **kwargs) -> Any:
        return self.env.render(*args, **kwargs)


class AddXpositionWrapper(gym.ObservationWrapper):

    def __init__(self, env: gym.wrappers.TimeLimit) -> None:
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            self.observation_space.low[0],
            self.observation_space.high[0],
            (self.observation_space.shape[0] + 1,)
        )
        self._max_episode_steps = self.env._max_episode_steps

    def observation(self, observation):
        return np.concatenate(
            [self.env.unwrapped.sim.data.qpos.flat[:1], observation], axis=0
        )


class NominalMuJoCoEnv(MuJoCoEnv):

    def __init__(self,
                 env: AddXpositionWrapper, # Offline MuJoCo environments with extended state
                 model_fit_seed: int,
                 model_fit_kwargs: Dict[str, Any] = {}
                 ) -> None:
        self.model_fit_seed = model_fit_seed
        self.model_fit_kwargs = model_fit_kwargs
        self.params = None
        super().__init__(env)

    def get_dataset(self) -> Any:
        """ Extend the observations in the dataset with qpos[0] from info.
        Use qpos[0] for the next observation in terminal states.
        """
        org_data = self.env.env.get_dataset()
        qpos = org_data["infos/qpos"][:, :1]
        terminal = np.logical_or(org_data["terminals"], org_data["timeouts"])
        next_qpos = np.concatenate([qpos[1:], qpos[-1:]], axis=0)
        next_qpos[terminal] = qpos[terminal]
        org_data["next_observations"] = np.concatenate([next_qpos, org_data["next_observations"]], axis=-1)
        org_data["observations"] = np.concatenate([qpos, org_data["observations"]], axis=-1)
        return org_data

    def fit_linear_model(self) -> jaxopt.OptStep:
        dataset = self.get_dataset()
        # Use only non-terminal data, since terminal next_observations are not valid.
        non_terminal = (1 - np.logical_or(dataset["terminals"],
                        dataset["timeouts"])[:-1]).astype("bool")
        state = dataset["observations"][:-1][non_terminal]
        next_state = dataset["next_observations"][:-1][non_terminal]
        action = dataset["actions"][:-1][non_terminal]

        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.shape[0]
        key = jax.random.PRNGKey(self.model_fit_seed)
        key, key_A, key_B = jax.random.split(key, 3)
        dynamics = {
            "A": jax.random.normal(key_A, (state_size, state_size)),
            "B": jax.random.normal(key_B, (state_size, action_size)),
        }

        def _l2_loss_on_dynamics(params, x, u, x_prime):
            residuals = params["A"] @ x.T + params["B"] @ u.T - x_prime.T
            return jnp.mean((residuals ** 2).sum(axis=0))

        solver = jaxopt.LBFGS(fun=_l2_loss_on_dynamics, maxiter=1000, **self.model_fit_kwargs)
        return solver.run(dynamics, x=state, u=action, x_prime=next_state)

    def nominal_model(self, *args, **kwargs) -> Any:
        if self.params is None:
            raise RuntimeError("Linear nominal model is not initialized!")
        return NominalLinearEnvParams(
            matrices=LinearDiscreteSystem(
                a_matrix=self.params["A"],
                b_matrix=self.params["B"],
                e_matrix=np.eye(self.state_size),
                c_matrix=np.eye(self.state_size),
                d_matrix=np.zeros((self.state_size, self.action_size))
            ),
            constraints=self.constraints,
            costs=None
        )


class Walker2dEnv(MuJoCoEnv):

    def __init__(self, add_x_pos: bool = False) -> None:
        env = gym.make("walker2d-medium-v2")
        if add_x_pos:
            env = AddXpositionWrapper(env)
        super().__init__(env)


class HalfCheetahEnv(MuJoCoEnv):

    def __init__(self, add_x_pos: bool = False) -> None:
        env = gym.make("halfcheetah-medium-v2")
        if add_x_pos:
            env = AddXpositionWrapper(env)
        super().__init__(env)


class HopperEnv(MuJoCoEnv):

    def __init__(self, add_x_pos: bool = False) -> None:
        env = gym.make("hopper-medium-v2")
        if add_x_pos:
            env = AddXpositionWrapper(env)
        super().__init__(env)


class OldSchoolWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reset(self, *args, **kwargs) -> np.ndarray:
        state, info = self.env.reset(*args, **kwargs)
        return state

    def step(self, action: np.ndarray) -> Tuple[Union[np.ndarray, float, bool, Dict[str, Any]]]:
        next_state, reward, termination, truncation, info = self.env.step(action)
        return next_state, reward, termination or truncation, info

    def get_normalized_score(self, score: float) -> float:
        return self.env.env.env.get_normalized_score(score)


class AugmentedStateWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, feature_handler: FeatureHandler, *args, **kwargs):
        self.feature_handler = feature_handler
        self._history = None
        self._past_state = None
        super().__init__(env, *args, **kwargs)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(self.feature_handler.aug_state_size,))

    def reset(self, *args, **kwargs):
        _state, *rest = self.env.reset(*args, **kwargs)
        self._history = self.feature_handler.reset_history()
        self._past_state = _state
        return self.feature_handler.augment_state(_state, self._history), *rest

    def step(self,
             action: np.ndarray
             ) -> Tuple[Union[np.ndarray, float, bool, Optional[Dict[str, Any]]]]:
        _next_state, *rest = self.env.step(action)
        self._history = self.feature_handler.update_history(
            state=self._past_state,
            action=action,
            next_state=_next_state,
            history=self._history)
        self._past_state = _next_state.copy()
        next_state = self.feature_handler.augment_state(_next_state, self._history)
        return next_state, *rest
