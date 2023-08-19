from typing import Any, Tuple, Dict, Optional, Union
import numpy as np
import gym
import d4rl

from io_agent.plant.base import (Plant,
                                 NominalLinearEnvParams,
                                 LinearConstraints,
                                 LinearConstraint)


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
    

class Walker2dEnv(MuJoCoEnv):

    def __init__(self) -> None:
        env = gym.make("walker2d-medium-v2")
        super().__init__(env)

class HalfCheetahEnv(MuJoCoEnv):

    def __init__(self) -> None:
        env = gym.make("halfcheetah-medium-v2")
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
