raise DeprecationWarning

import unittest
import numpy as np

from io_agent.plant.fighter import FighterEnv
from io_agent.plant.base import InputValues
from io_agent.evaluator import ControlLoop
from io_agent.control.mpc import MPC
from io_agent.control.rmpc import RobustMPC


class TestRobustMPC(unittest.TestCase):

    def setUp(self) -> None:
        self.env_length = 61
        self.horizon = 20
        self.disturbance_bias = np.array([0.0, 0.005]).reshape(-1, 1)

        seed_rng = np.random.default_rng(42)

        self.plant = FighterEnv(
            max_length=self.env_length,
            disturbance_bias=self.disturbance_bias,
            rng=np.random.default_rng(seed_rng.integers(0, 2**30)))
        # plant = LinearizationWrapper(plant)
        self.mpc_agent = MPC(
            action_size=self.plant.action_size,
            state_size=self.plant.state_size,
            noise_size=self.plant.noise_size,
            output_size=self.plant.output_size,
            horizon=self.horizon)
        self.mpc_agent.optimizer = self.mpc_agent.prepare_optimizer(
            self.plant.nominal_model(
                lin_point=InputValues(
                    state=np.zeros((self.plant.state_size,)),
                    action=np.zeros((self.plant.action_size,)),
                    noise=np.zeros((self.plant.noise_size,)),
                )
            ))

        evaluator = ControlLoop(
            plant=self.plant,
            controller=self.mpc_agent,
            state_disturbance=self.plant.disturbances.state,
            action_disturbance=self.plant.disturbances.action,
            output_disturbance=self.plant.disturbances.output,
            rng=np.random.default_rng(seed_rng.integers(0, 2**30))
        )
        self.mpc_trajectory = evaluator.simulate(
            initial_state=None,
            use_foresight=True,
        )

    def test_zero_rho(self):
        """ RMPC and MPC must give the same actions for the same input when rho is set to zero
        """
        rmpc_agent = RobustMPC(action_size=self.plant.action_size,
                               state_size=self.plant.state_size,
                               noise_size=self.plant.noise_size,
                               output_size=self.plant.output_size,
                               horizon=self.horizon,
                               rho=0.0,
                               state_constraints_flag=True,
                               input_constraints_flag=True,)
        rmpc_agent.optimizer = rmpc_agent.prepare_optimizer(
            self.plant.nominal_model(
                lin_point=InputValues(
                    state=np.zeros((self.plant.state_size,)),
                    action=np.zeros((self.plant.action_size,)),
                    noise=np.zeros((self.plant.noise_size,)),
                )
            ))

        for step, transition in enumerate(self.mpc_trajectory):
            action, min_cost = rmpc_agent.compute(
                initial_state=transition.state,
                reference_sequence=self.plant.reference_sequence[:, step: step + self.horizon],
                state_disturbance=self.plant.disturbances.state[:, step: step + self.horizon]
            )
            action = np.clip(action,
                             self.plant.action_space.low,
                             self.plant.action_space.high)
            diff_norm = np.linalg.norm(action - transition.action).item()
            self.assertTrue(np.allclose(action, transition.action, atol=1e-2),
                            f"Difference at step: {step} with error: {diff_norm}")
