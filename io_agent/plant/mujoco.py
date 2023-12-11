from typing import Any, Tuple, Dict, Optional, Union
import numpy as np
import gym
import d4rl

from io_agent.plant.base import (Plant,
                                 NominalLinearEnvParams,
                                 LinearConstraints,
                                 LinearConstraint)
from io_agent.utils import FeatureHandler


class MuJoCoEnv(Plant):

    def __init__(self, env: gym.Env) -> None:
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


class Walker2dEnv(MuJoCoEnv):
    task_name: str = "walker2d-medium-v2"

    def __init__(self) -> None:
        env = gym.make(self.task_name)
        super().__init__(env)


class HalfCheetahEnv(MuJoCoEnv):
    task_name: str = "halfcheetah-medium-v2"

    def __init__(self) -> None:
        env = gym.make(self.task_name)
        super().__init__(env)


class HopperEnv(MuJoCoEnv):
    task_name: str = "hopper-medium-v2"

    def __init__(self) -> None:
        env = gym.make(self.task_name)
        super().__init__(env)


class OldSchoolWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reset(self, *args, **kwargs) -> np.ndarray:
        state, info = self.env.reset(*args, **kwargs)
        return state

    def step(self, action: np.ndarray) -> Tuple[Union[np.ndarray, float, bool, Dict[str, Any]]]:
        next_state, reward, termination, truncation, info = self.env.step(
            action)
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
        next_state = self.feature_handler.augment_state(
            _next_state, self._history)
        return next_state, *rest
