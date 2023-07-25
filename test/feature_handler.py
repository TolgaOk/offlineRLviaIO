import unittest
import numpy as np

from io_agent.plant.fighter import FighterEnv
from io_agent.evaluator import ControlLoop
from io_agent.control.mpc import MPC
from io_agent.control.io import IOController, AugmentDataset
from io_agent.utils import FeatureHandler


class TestFeatureHandler(unittest.TestCase):

    def setUp(self) -> None:
        self.seed_rng = np.random.default_rng()
        self.horizon = 5
        self.n_past = 3
        self.env_length = 51
        disturbance_bias = False

        plant = FighterEnv(max_length=self.env_length,
                           disturbance_bias=disturbance_bias,
                           rng=np.random.default_rng(self.seed_rng.integers(0, 2**30)))
        self.expert_agent = MPC(
            action_size=plant.action_size,
            state_size=plant.state_size,
            noise_size=plant.noise_size,
            output_size=plant.output_size,
            horizon=self.horizon)
        self.expert_agent.optimizer = self.expert_agent.prepare_optimizer(plant.env_params)

        state_disturbance = plant.state_disturbance.copy()
        self.control_loop = ControlLoop(
            state_disturbance=state_disturbance,
            output_disturbance=plant.output_disturbance,
            plant=plant,
            controller=self.expert_agent,
            rng=np.random.default_rng(self.seed_rng.integers(0, 2**30))
        )

        feature_handler = FeatureHandler(
            env_params=plant.env_params,
            n_past=self.n_past,
            add_bias=True,
            use_action_regressor=False,
            use_noise_regressor=True,
            use_state_regressor=False)
        self.io_agent = IOController(
            env_params=plant.env_params,
            include_constraints=True,
            soften_state_constraints=True,
            state_constraints_flag=True,
            action_constraints_flag=True,
            dataset_length=self.env_length - (self.horizon + self.n_past),
            feature_handler=feature_handler)
        self.augmenter = AugmentDataset(
            expert_agent=self.expert_agent,
            feature_handler=feature_handler,
        )

    def test_expert_actions(self) -> None:

        trajectories = [self.control_loop.simulate(
            initial_state=None,
            use_foresight=True,
        )]
        augmented_trajectories = self.augmenter(trajectories)

        for aug_tran, tran in zip(augmented_trajectories, trajectories[0][self.n_past:]):
            self.assertTrue(np.allclose(aug_tran.expert_action, tran.action, atol=1e-7))

    def test_compute_and_train_discrepancy(self) -> None:
        trajectories = [self.control_loop.simulate(
            initial_state=None,
            use_foresight=True,
        )]
        self.io_agent.reset()

        aug_trajectories = self.augmenter(trajectories)

        self.io_agent.train(aug_trajectories, rng=np.random.default_rng())
        self.io_agent.action_optimizer = self.io_agent.prepare_action_optimizer()

        l2_losses = []

        for index in range(len(trajectories[0]) - self.expert_agent.horizon):
            tran = trajectories[0][index]
            io_action = self.io_agent.compute(tran.state, None)[0]
            self.io_agent._past_action = tran.action

            if index >= self.n_past:
                aug_tran = aug_trajectories[index - self.n_past]
                expert_action = aug_tran.expert_action
                self.assertTrue(np.allclose(
                    aug_tran.aug_state,
                    self.io_agent.feature_handler.augment_state(tran.state, self.io_agent._history),
                    atol=1e-7
                ))
                l2_losses.append(np.linalg.norm(expert_action - io_action, ord=2))

        self.assertTrue(np.mean(l2_losses) < 1e-1)
