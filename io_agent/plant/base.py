from typing import Optional, Dict, Union, Tuple
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
    noise: np.ndarray


@dataclass
class SystemInput():
    state: sympy.Matrix
    action: sympy.Matrix
    noise: sympy.Matrix
    state_dot: sympy.Matrix


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
    nonlinear_dyn: sympy.Matrix
    nonlinear_out: sympy.Matrix


@dataclass
class DiscreteLinearEnvMatrices():
    a_matrix: np.ndarray
    b_matrix: np.ndarray
    e_matrix: np.ndarray
    c_matrix: np.ndarray
    d_matrix: np.ndarray
    lin_input: InputValues
    nonlinear_dyn: np.ndarray
    nonlinear_out: np.ndarray


@dataclass
class NominalLinearEnvParams():
    """ Parameters of a Nominal Linear time invariant system
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
    matrices: DiscreteLinearEnvMatrices
    constraints: LinearConstraints
    costs: QuadraticCosts


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

    def nominal_model(self, lin_point: InputValues, discretization_method: str = "exact") -> NominalLinearEnvParams:
        linear_system = self.linearize(self.symbolic_dynamical_system())
        discerete_matrices = self.discretize(
            linear_system,
            lin_point=lin_point,
            method=discretization_method)
        return NominalLinearEnvParams(
            matrices=discerete_matrices,
            constraints=self.constraints,
            costs=self.costs
        )

    @abstractmethod
    def symbolic_dynamical_system(self) -> DynamicalSystem:
        raise NotImplementedError

    @abstractmethod
    def fill_symbols(self,
                     input_values: InputValues,
                     dynamical_sys: LinearSystem
                     ) -> Dict[str, Union[float, np.ndarray]]:
        raise NotImplementedError

    def linearize(self, dynamical_sys: DynamicalSystem) -> LinearSystem:
        state = dynamical_sys.sys_input.state
        action = dynamical_sys.sys_input.action
        noise = dynamical_sys.sys_input.noise
        next_state = dynamical_sys.sys_input.state_dot

        jac_dyn_a = dynamical_sys.dyn_eq.jacobian(state)
        jac_dyn_b = dynamical_sys.dyn_eq.jacobian(action)
        jac_dyn_e = dynamical_sys.dyn_eq.jacobian(noise)
        jac_out_c = dynamical_sys.out_eq.jacobian(next_state)
        jac_out_d = dynamical_sys.out_eq.jacobian(action)

        return LinearSystem(
            a_matrix=jac_dyn_a,
            b_matrix=jac_dyn_b,
            e_matrix=jac_dyn_e,
            c_matrix=jac_out_c,
            d_matrix=jac_out_d,
            nonlinear_dyn=dynamical_sys.dyn_eq,
            nonlinear_out=dynamical_sys.out_eq,
        )

    def _evaluate_sym(self, sym: sympy.Matrix, values: Dict[str, Union[np.ndarray, float]]) -> Union[np.ndarray, float]:
        return np.array(sym.evalf(subs=values), dtype=np.float64)

    def discretize(self,
                   dynamical_sys: LinearSystem,
                   lin_point: InputValues,
                   method: str = "exact"
                   ) -> DiscreteLinearEnvMatrices:
        values = self.fill_symbols(lin_point, dynamical_sys)
        continouos_matrices = {key: self._evaluate_sym(getattr(dynamical_sys, f"{key}_matrix"), values)
                               for key in ("a", "b", "c", "e", "d")}
        x_dot_zero = self._evaluate_sym(dynamical_sys.nonlinear_dyn, values).flatten()
        y_zero = self._evaluate_sym(dynamical_sys.nonlinear_out, values)

        if method == "exact":
            method_fn = self._exact_discretize
        elif method == "euler":
            method_fn = self._euler_discretize
        else:
            raise ValueError(f"Unknown discretization method: {method}")
        discrete_matrices = method_fn(continouos_matrices, x_dot_zero, dt=values["dt"])
        constants = discrete_matrices.pop("constants")

        return DiscreteLinearEnvMatrices(
            **discrete_matrices,
            lin_input=lin_point,
            nonlinear_dyn=constants,
            nonlinear_out=y_zero.flatten()
        )

    def _exact_discretize(self,
                          continouos_matrices: Dict[str, np.ndarray],
                          x_dot_zero: np.ndarray,
                          dt: float,
                          ) -> Dict[str, np.ndarray]:
        n_state = continouos_matrices["a"].shape[0]
        n_action = continouos_matrices["b"].shape[1]
        n_noise = continouos_matrices["e"].shape[1]

        n_size = n_state * 2 + n_action + n_noise
        matrix = np.zeros((n_size, n_size))
        matrix[:n_state, :n_state] = continouos_matrices["a"]
        matrix[:n_state, n_state: 2 * n_state] = np.eye(n_state)
        matrix[:n_state, 2 * n_state:-n_noise] = continouos_matrices["b"]
        matrix[:n_state, 2 * n_state + n_action:] = continouos_matrices["e"]

        exp_matrix = scipy.linalg.expm(matrix * dt)
        discerete_a_matrix = exp_matrix[:n_state, :n_state]
        discerete_b_matrix = exp_matrix[:n_state, 2 * n_state:-n_noise]
        exp_a = exp_matrix[:n_state, n_state: 2 * n_state]
        discerete_e_matrix = exp_matrix[:n_state, 2 * n_state + n_action:]

        return dict(
            a_matrix=discerete_a_matrix,
            b_matrix=discerete_b_matrix,
            e_matrix=discerete_e_matrix,
            c_matrix=continouos_matrices["c"],
            d_matrix=continouos_matrices["d"],
            constants=(exp_a @ x_dot_zero)
        )

    def _euler_discretize(self,
                          continouos_matrices: Dict[str, np.ndarray],
                          dt: float,
                          *args,
                          ) -> Dict[str, np.ndarray]:
        raise NotImplementedError
        return dict(
            a_matrix=np.eye(continouos_matrices["a"].shape[0]) + continouos_matrices["a"] * dt,
            b_matrix=continouos_matrices["b"] * dt,
            e_matrix=continouos_matrices["e"] * dt,
            c_matrix=continouos_matrices["c"],
            d_matrix=continouos_matrices["d"],
        )
