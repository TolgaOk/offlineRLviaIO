from typing import Tuple, Union, Optional, Type, Dict, Any
import numpy as np

try:
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3 import PPO
except ImportError as err:
    raise ImportError("RL agents require StableBaselines3!$ pip install stable_baselines_3")

from io_agent.plant.base import Plant


class PPoController():

    def __init__(self,
                 model_or_path: Union[str, PPO],
                 ) -> None:
        self.model = model_or_path
        if isinstance(model_or_path, str):
            self.model = PPO.load(model_or_path)
        self.horizon = 0  # Required by ControlLoop

    def compute(self,
                state: np.ndarray,
                reference_sequence: Optional[np.ndarray] = None,
                ) -> Tuple[Union[np.ndarray, None]]:
        return self.model.predict(
            state,
            state=None,
            episode_start=None,
            deterministic=True,
        )

    @staticmethod
    def train(env_fn: Type[Plant],
              n_envs: int,
              seed: int,
              path: Optional[str] = None,
              total_timesteps: int = int(2e5),
              verbose: bool = True,
              ppo_kwargs: Dict[str, Any] = {},
              **learn_kwargs
              ) -> PPO:
        vec_env = make_vec_env(env_fn, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
        model = PPO("MlpPolicy", vec_env, verbose=int(verbose), seed=seed, **ppo_kwargs)
        model.learn(total_timesteps=total_timesteps, progress_bar=False, **learn_kwargs)
        if path is not None:
            model.save(path)
        return model

    def reset(self) -> None:
        pass
