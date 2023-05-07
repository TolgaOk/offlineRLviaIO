from typing import Optional, Any, Dict, List
import numpy as np
from dataclasses import dataclass

from io_agent.plant.base import Plant
from io_agent.control.mpc import Optimizer, MPC

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
                 state_disturbance: np.ndarray,
                 output_disturbance: np.ndarray,
                 plant: Plant,
                 controller: MPC,
                 ) -> None:
        self.state_disturbance = state_disturbance
        self.output_disturbance = output_disturbance
        self.plant = plant
        self.controller = controller

    def simulate(self,
                 initial_state: Optional[np.ndarray],
                 use_foresight: bool
                 ) -> List[Transition]:
        """ Simulate a single episode/trajectory starting from the given initial
        state

        Args:
            initial_state (Optional[np.ndarray]): Initial state of shape (S,)
            use_foresight (bool): If true; provide future state and output disturbances
                to the MPC agent

        Returns:
            List[Transition]: List of environment transitions
        """

        simulation_sequence = []

        if initial_state is None:
            initial_state, _ = self.plant.reset()

        state = initial_state
        done = False
        step = 0
        while not done:
            if use_foresight:
                action, min_cost = self.controller.compute(
                    initial_state=state,
                    reference_sequence = self.plant.reference_sequence[:, step: step + self.controller.horizon],
                    output_disturbance=self.output_disturbance[:, step: step +
                                                               self.controller.horizon],
                    state_disturbance=self.state_disturbance[:, step: step + self.controller.horizon],
                )
            else:
                action, min_cost = self.controller.compute(
                    initial_state=state,
                    reference_sequence = self.plant.reference_sequence[:, step: step + self.controller.horizon],
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
