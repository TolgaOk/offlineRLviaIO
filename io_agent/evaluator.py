from typing import Any, Dict, List
import numpy as np
from dataclasses import dataclass

from io_agent.plant.base import Plant
from io_agent.control.mpc import MPC


@dataclass
class Transition:
    state: np.ndarray
    action: np.ndarray
    next_state: np.ndarray
    cost: float
    termination: bool
    truncation: bool
    info: Dict[str, Any]


class ControlLoop():
    """ Run the plant/environment with the actions/inputs of the MPC agent.

    Args:
        state_disturbance (np.ndarray): The state disturbance array of shape (W, T)
            where T denotes the environment length and W denotes the state disturbance
            size.
        output_disturbance (np.ndarray): The output disturbance array of shape (S, T)
            where T denotes the environment length and S denotes the state space size.
        plant (Plant): Environment to simulate
        controller (MPC): MPC agent
    """

    def __init__(self,
                 plant: Plant,
                 controller: MPC,
                 rng: np.random.Generator
                 ) -> None:
        self.plant = plant
        self.controller = controller
        self.rng = rng

    def simulate(self,
                 bias_aware: bool,
                 use_foresight: bool
                 ) -> List[Transition]:
        """ Simulate a single episode/trajectory starting from the given initial
        state

        Args:
            use_foresight (bool): If true; provide future state and output disturbances
                to the MPC agent

        Returns:
            List[Transition]: List of environment transitions
        """

        simulation_sequence = []

        initial_state, info = self.plant.reset(
            seed=self.rng.integers(0, 2**30).item(),
            options=dict(bias_aware=bias_aware))
        horizon = self.controller.horizon if self.controller.horizon is not None else 0
        disturbance = info["disturbance"]

        self.controller.reset()
        state = initial_state
        done = False
        step = 0
        while not done:
            if use_foresight:
                action, _ = self.controller.compute(
                    initial_state=state,
                    reference_sequence=self.plant.reference_sequence[:, step: step + horizon],
                    output_disturbance=disturbance.output[:, step: step + horizon],
                    state_disturbance=disturbance.state[:, step: step + horizon],
                    action_disturbance=disturbance.action[:, step: step + horizon],
                )
            else:
                action, _ = self.controller.compute(
                    state,
                    self.plant.reference_sequence[:, step: step + horizon],
                )
            action = np.clip(action,
                             self.plant.action_space.low,
                             self.plant.action_space.high)
            next_state, cost, termination, truncation, info = self.plant.step(action)
            done = termination or truncation
            transition = Transition(
                state=state,
                action=action,
                next_state=next_state,
                cost=cost,
                termination=termination,
                truncation=truncation,
                info=info
            )
            simulation_sequence.append(transition)
            state = next_state
            step += 1

        return simulation_sequence
