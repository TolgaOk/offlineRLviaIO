from typing import Optional, Dict, Union
import numpy as np
from dataclasses import dataclass, asdict
from abc import abstractmethod
import gymnasium as gym
import sympy
import scipy


@dataclass
class QuadraticCosts():
    state: np.ndarray
    action: np.ndarray
    final: np.ndarray


@dataclass
class LinearConstraints:
    state_constraint_matrix: np.ndarray
    state_constraint_vector: np.ndarray
    action_constraint_matrix: np.ndarray
    action_constraint_vector: np.ndarray


@dataclass
class InputValues():
    state: np.ndarray
    action: np.ndarray
    noise: Optional[np.ndarray] = None


@dataclass
class SystemInput():
    state: sympy.Matrix
    action: sympy.Matrix
    noise: sympy.Matrix


@dataclass
class DynamicalSystem():
    """ General dyanmical system
        \dot{x}(t) = f(x(t), u(t), w(t))
        y(t) = g(x(t), x(t))
    """
    sys_input: SystemInput
    dyn_eq: sympy.Matrix
    out_eq: sympy.Matrix


@dataclass
class LinearSystem():
    """ Linear system matrices that yield following linear equations
        \dot{x}(t) = Ax(t) + Bu(t) + Ew(t)
        y(t) = Cx(t) + Du(t)
    """
    a_matrix: sympy.Matrix
    b_matrix: sympy.Matrix
    e_matrix: sympy.Matrix
    c_matrix: sympy.Matrix
    d_matrix: sympy.Matrix


@dataclass
class DiscreteLinearEnvMatrices():
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
    d_matrix: np.ndarray


@dataclass
class EnvMatrices():
    a_matrix: np.ndarray
    b_matrix: np.ndarray
    e_matrix: np.ndarray
    c_matrix: np.ndarray
    d_matrix: np.ndarray
    state_constraint_matrix: np.ndarray
    state_constraint_vector: np.ndarray
    action_constraint_matrix: np.ndarray
    action_constraint_vector: np.ndarray
    state_cost: np.ndarray
    action_cost: np.ndarray
    final_cost: np.ndarray


class Plant(gym.Env):

    state_size: int
    action_size: int
    noise_size: int
    ref_size: int
    output_size: int

    def __init__(self,
                 costs: QuadraticCosts,
                 constraints: Optional[LinearConstraints] = None,
                 reference_sequence: Optional[np.ndarray] = None
                 ) -> None:
        """ Base Plant class with quadratic costs and optinal linear constraints and reference_sequence
        """
        self.costs = costs
        self.constraints = constraints
        self.reference_sequence = reference_sequence
        super().__init__()

    @abstractmethod
    def symbolic_dynamical_system(self) -> DynamicalSystem:
        raise NotImplementedError

    @abstractmethod
    def fill_symbols(self, input_values: InputValues) -> Dict[str, Union[float, np.ndarray]]:
        raise NotImplementedError

    def linearize(self, dynamical_sys: DynamicalSystem) -> LinearSystem:

        init_state = dynamical_sys.sys_input.state
        init_action = dynamical_sys.sys_input.action
        init_noise = dynamical_sys.sys_input.noise

        jac_dyn_a = dynamical_sys.dyn_eq.jacobian(init_state)
        jac_dyn_b = dynamical_sys.dyn_eq.jacobian(init_action)
        jac_dyn_e = dynamical_sys.dyn_eq.jacobian(init_noise)

        n_state = init_state.shape[0]
        n_action = init_action.shape[0]
        n_noise = init_noise.shape[0]

        a_matrix = sympy.Matrix(
            [[jac_dyn_a,
              -sympy.diag(*(jac_dyn_a @ init_state))
              - sympy.diag(*(jac_dyn_b @ init_action))
              - sympy.diag(*(jac_dyn_e @ init_noise))
              + sympy.diag(*dynamical_sys.dyn_eq)
              ],
             [sympy.zeros(n_state, n_state), sympy.eye(n_state)]])
        b_matrix = sympy.Matrix(
            [[jac_dyn_b],
             [sympy.zeros(n_state, n_action)]])
        e_matrix = sympy.Matrix(
            [[jac_dyn_e],
             [sympy.zeros(n_state, n_noise)]])

        jac_out_c = dynamical_sys.out_eq.jacobian(init_state)
        jac_out_d = dynamical_sys.out_eq.jacobian(init_action)

        c_matrix = sympy.Matrix(
            [[jac_out_c,
              - sympy.diag(*(jac_out_c @ init_state))
              - sympy.diag(*(jac_out_d @ init_action))
              + sympy.diag(*dynamical_sys.out_eq)],
             [sympy.zeros(n_state, 2 * n_state)]])
        d_matrix = sympy.Matrix(
            [[jac_out_d],
             [sympy.zeros(n_state, n_action)]]
        )

        return LinearSystem(
            a_matrix=a_matrix,
            b_matrix=b_matrix,
            e_matrix=e_matrix,
            c_matrix=c_matrix,
            d_matrix=d_matrix)

    def discretize(self,
                   dynamical_sys: LinearSystem,
                   lin_point: InputValues,
                   method: str = "exact"
                   ) -> DiscreteLinearEnvMatrices:
        if method == "exact":
            return self._exact_discretize(dynamical_sys, lin_point)
        if method == "euler":
            return self._euler_discretize(dynamical_sys, lin_point)
        else:
            raise ValueError(f"Unknown discretization method: {method}")

    def _exact_discretize(self,
                          dynamical_sys: LinearSystem,
                          lin_point: InputValues,
                          ) -> DiscreteLinearEnvMatrices:
        n_state = dynamical_sys.a_matrix.shape[0]
        n_action = dynamical_sys.b_matrix.shape[1]
        n_noise = dynamical_sys.e_matrix.shape[1]

        values = self.fill_symbols(lin_point)
        cont_a_matrix = np.array(dynamical_sys.a_matrix.evalf(subs=values), dtype=np.float64)
        cont_b_matrix = np.array(dynamical_sys.b_matrix.evalf(subs=values), dtype=np.float64)
        cont_e_matrix = np.array(dynamical_sys.e_matrix.evalf(subs=values), dtype=np.float64)
        cont_c_matrix = np.array(dynamical_sys.c_matrix.evalf(subs=values), dtype=np.float64)
        cont_d_matrix = np.array(dynamical_sys.d_matrix.evalf(subs=values), dtype=np.float64)

        matrix = np.zeros((n_state + n_action + n_noise, n_state + n_action + n_noise))
        matrix[:n_state, :n_state] = cont_a_matrix
        matrix[:n_state, n_state:-n_noise] = cont_b_matrix
        matrix[:n_state, n_state + n_action:] = cont_e_matrix

        dt = values["dt"]
        exp_matrix = scipy.linalg.expm(matrix * dt)
        discerete_a_matrix = exp_matrix[:n_state, :n_state]
        discerete_b_matrix = exp_matrix[:n_state, n_state:-n_noise]
        discerete_e_matrix = exp_matrix[:n_state, n_state + n_noise:]

        return DiscreteLinearEnvMatrices(
            a_matrix=discerete_a_matrix,
            b_matrix=discerete_b_matrix,
            e_matrix=discerete_e_matrix,
            c_matrix=cont_c_matrix,
            d_matrix=cont_d_matrix,
        )

    def _euler_discretize(self,
                          dynamical_sys: LinearSystem,
                          lin_point: InputValues,
                          ) -> DiscreteLinearEnvMatrices:
        n_state = dynamical_sys.a_matrix.shape[0]
        n_action = dynamical_sys.b_matrix.shape[1]
        n_noise = dynamical_sys.e_matrix.shape[1]

        values = self.fill_symbols(lin_point)
        cont_a_matrix = np.array(dynamical_sys.a_matrix.evalf(subs=values), dtype=np.float64)
        cont_b_matrix = np.array(dynamical_sys.b_matrix.evalf(subs=values), dtype=np.float64)
        cont_e_matrix = np.array(dynamical_sys.e_matrix.evalf(subs=values), dtype=np.float64)
        cont_c_matrix = np.array(dynamical_sys.c_matrix.evalf(subs=values), dtype=np.float64)
        cont_d_matrix = np.array(dynamical_sys.d_matrix.evalf(subs=values), dtype=np.float64)

        dt = values["dt"]
        return DiscreteLinearEnvMatrices(
            a_matrix=np.eye(n_state) + cont_a_matrix * dt,
            b_matrix=cont_b_matrix * dt,
            e_matrix=cont_e_matrix * dt,
            c_matrix=cont_c_matrix,
            d_matrix=cont_d_matrix,
        )
