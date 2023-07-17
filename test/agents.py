from typing import List, Optional, Dict, Union, Any, Callable
import unittest
import numpy as np
from tqdm.notebook import tqdm
from itertools import chain
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
from multiprocessing import Pool
from functools import partial
import cvxpy as cp

from io_agent.plant.fighter import FighterEnv, fighter_env_params
from io_agent.trainer import ControlLoop, Transition
from io_agent.control.mpc import MPC
from io_agent.control.rmpc import RobustMPC
from io_agent.control.io import IOController
from io_agent.utils import FeatureHandler


class TestFeatureHandler(unittest.TestCase):

    def setUp(self):
        self.seed_rng = np.random.default_rng()
        self.horizon = 5
        self.n_past = 3
        self.env_length = 51
        disturbance_bias = False

        self.expert_agent = MPC(
            env_params=fighter_env_params,
            horizon=self.horizon)

        plant = FighterEnv(max_length=self.env_length,
                           env_params=fighter_env_params,
                           disturbance_bias=disturbance_bias,
                           rng=np.random.default_rng(self.seed_rng.integers(0, 2**30)))
        state_disturbance = plant.state_disturbance.copy()
        self.trainer = ControlLoop(
            state_disturbance=state_disturbance,
            output_disturbance=plant.output_disturbance,
            plant=plant,
            controller=self.expert_agent,
            rng=np.random.default_rng(self.seed_rng.integers(0, 2**30))
        )

        feature_handler = FeatureHandler(
            env_params=fighter_env_params,
            n_past=self.n_past,
            add_bias=True,
            use_action_regressor=False,
            use_noise_regressor=True,
            use_state_regressor=False)
        self.io_agent = IOController(
            env_params=fighter_env_params,
            expert_agent=self.expert_agent,
            include_constraints=True,
            soften_state_constraints=True,
            state_constraints_flag=True,
            action_constraints_flag=True,
            dataset_length=self.env_length - (self.horizon + self.n_past),
            feature_handler=feature_handler)

    def test_expert_actions(self):

        trajectories = self.trainer.simulate(
            initial_state=None,
            use_foresight=True,
        )
        augmented_trajectories = self.io_agent.augment_dataset([trajectories])

        for aug_tran, tran in zip(augmented_trajectories, trajectories[self.n_past:]):
            self.assertTrue(np.allclose(aug_tran.expert_action, tran.action, atol=1e-7))

    def test_compute_and_train_discrepancy(self):
        trajectories = [self.trainer.simulate(
            initial_state=None,
            use_foresight=True,
        )]
        self.io_agent.reset()

        aug_trajectories = self.io_agent.augment_dataset(trajectories)
        self.io_agent.train(trajectories, rng=np.random.default_rng())
        self.io_agent.action_optimizer = self.io_agent.prepare_action_optimizer()

        l2_losses = []

        for index in range(len(trajectories[0]) - self.io_agent.expert_agent.horizon):
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
