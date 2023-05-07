import numpy as np
import gymnasium as gym
from dataclasses import dataclass


@dataclass
class LinearEnvParams():
    """ Parameters of a Linear time invariant system
        system  x' = Ax + Bu + Ew
                o = Cx
        cost    (o-r)^TQ_{x}(o-r) + u^TQ_{u}u 
        s.t.    G_{x}x <= h_x
                G_{u}u <= h_u

        Where x is the state of the system, o denotes the output, r denotes
        the reference, Q_x denotes the state_cost (or final state cost),
        Q_u denotes the action cost, G_{x} denotes the state constraints matrix,
        h_x denotes the state constraints, G_{u} denotes the action constraints
        matrix, h_u denotes the action constraints and w denotes the state
        disturbance, and u denotes the action/input.
    """
    a_matrix: np.ndarray
    b_matrix: np.ndarray
    e_matrix: np.ndarray
    c_matrix: np.ndarray
    state_cost: np.ndarray
    action_cost: np.ndarray
    final_cost: np.ndarray
    state_constraint_matrix: np.ndarray
    state_constraint_vector: np.ndarray
    action_constraint_matrix: np.ndarray
    action_constraint_vector: np.ndarray


class Plant(gym.Env):

    def __init__(self, params: LinearEnvParams, reference_sequence: np.ndarray) -> None:
        """ Base Plant class

        Args:
            params (LinearEnvParams): Parameters that determines the behavior of the environment.
            reference_sequence (np.ndarray): Reference sequence array of shape (S, L)
                where S denotes the output/state size and L denotes the number of simulation step of the environment
        """
        self.params = params
        self.reference_sequence = reference_sequence
        super().__init__()
