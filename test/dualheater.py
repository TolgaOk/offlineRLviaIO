raise DeprecationWarning

import unittest
from copy import deepcopy

from io_agent.plant.dualheater import DualHeaterEnv, Constants
from io_agent.plant.base import LinearizationWrapper, InputValues, NominalLinearEnvParams
from dataclasses import asdict
import numpy as np


class TestDualHeater(unittest.TestCase):

    def setUp(self):
        env = DualHeaterEnv(60, None)
        self.env = LinearizationWrapper(env)

    def _nominal_env(self,
                     nom_model: NominalLinearEnvParams,
                     state: np.ndarray,
                     action: np.ndarray,
                     noise: np.ndarray,
                     **kwargs) -> np.ndarray:
        return (nom_model.matrices.a_matrix @ state +
                nom_model.matrices.b_matrix @ action +
                nom_model.matrices.e_matrix @ noise)

    def _test_linearization(self, state: np.ndarray, action: np.ndarray, noise: np.ndarray, **kwargs) -> None:
        point = InputValues(
            state=state,
            action=action,
            noise=noise,
            output_noise=np.zeros((self.env.output_size,))
        )
        nom_model = self.env.nominal_model(point)
        point.state = np.concatenate([point.state, np.ones_like(point.state)])
        linear_estimated_next_state = self._nominal_env(nom_model, **asdict(point))

        self.env.reset(options=dict(bias_aware=False))
        self.env.disturbances.state[:, 0] = point.noise
        self.env.disturbances.output[:, 0] = point.output_noise
        self.env.disturbances.action[:, 0] = 0
        self.env.env.state = point.state[:4]
        ode_next_state = self.env.step(point.action)[0]

        self.assertTrue(np.allclose(linear_estimated_next_state, ode_next_state, atol=1e-5),
                        "Linearized solution is not consistent with the ode solution at the point of linearization!")

    def test_default_linearization(self) -> None:
        lin_point = self.env.default_lin_point()
        self._test_linearization(**asdict(lin_point))

    def test_action_linearization(self) -> None:
        lin_point = self.env.default_lin_point()
        lin_point.action = np.random.uniform(
            low=self.env.action_space.low + 0.5, high=self.env.action_space.high - 0.5)
        self._test_linearization(**asdict(lin_point))

    def test_random_linearization(self) -> None:
        lin_point = self.env.default_lin_point()
        lin_point.action = np.random.uniform(
            low=self.env.action_space.low + 0.5, high=self.env.action_space.high - 0.5)
        lin_point.state = np.random.uniform(
            low=Constants.c2k - 10, high=Constants.c2k + 50, size=lin_point.state.shape[0])
        self._test_linearization(**asdict(lin_point))
