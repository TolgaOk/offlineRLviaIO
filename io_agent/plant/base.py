from typing import Optional, Dict, Union, Tuple, Any
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
class AffineSystem():
    """ Linear system matrices that yield following linear equations
        \dot{x}(t) = Ax(t) + Bu(t) + Ew(t) + K
        y(t) = Cx(t) + Du(t) + L
        where K and L are constant vectors.
    """
    a_matrix: sympy.Matrix
    b_matrix: sympy.Matrix
    e_matrix: sympy.Matrix
    c_matrix: sympy.Matrix
    d_matrix: sympy.Matrix
    dyn_constant: sympy.Matrix
    out_constant: sympy.Matrix


@dataclass
class AffineDiscreteSystem():
    a_matrix: np.ndarray
    b_matrix: np.ndarray
    e_matrix: np.ndarray
    c_matrix: np.ndarray
    d_matrix: np.ndarray
    dyn_constant: np.ndarray
    out_constant: np.ndarray


@dataclass
class LinearDiscreteSystem():
    a_matrix: np.ndarray
    b_matrix: np.ndarray
    e_matrix: np.ndarray
    c_matrix: np.ndarray
    d_matrix: np.ndarray


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
    matrices: LinearDiscreteSystem
    constraints: LinearConstraints
    costs: QuadraticCosts


class Plant(gym.Env):
    state_size: int
    action_size: int
    noise_size: int
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
    def fill_symbols(self,
                     input_values: InputValues,
                     dynamical_sys: DynamicalSystem
                     ) -> Dict[str, Union[float, np.ndarray]]:
        raise NotImplementedError

    def nominal_model(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def linearization(self,
                      lin_point: InputValues,
                      discretization_method: str = "exact"
                      ) -> NominalLinearEnvParams:
        affline_sys = self.affinization(self.symbolic_dynamical_system())
        affine_dsc_sys = self.discretization(
            affline_sys,
            lin_point=lin_point,
            method=discretization_method)
        return self.augmentation(
            affine_dsc_sys,
            costs=self.costs,
            constraints=self.constraints
        )

    def affinization(self, dynamical_sys: DynamicalSystem) -> AffineSystem:
        state = dynamical_sys.sys_input.state
        action = dynamical_sys.sys_input.action
        noise = dynamical_sys.sys_input.noise

        jac_dyn_a = dynamical_sys.dyn_eq.jacobian(state)
        jac_dyn_b = dynamical_sys.dyn_eq.jacobian(action)
        jac_dyn_e = dynamical_sys.dyn_eq.jacobian(noise)
        jac_out_c = dynamical_sys.out_eq.jacobian(state)
        jac_out_d = dynamical_sys.out_eq.jacobian(action)

        return AffineSystem(
            a_matrix=jac_dyn_a,
            b_matrix=jac_dyn_b,
            e_matrix=jac_dyn_e,
            c_matrix=jac_out_c,
            d_matrix=jac_out_d,
            dyn_constant=dynamical_sys.dyn_eq - (
                jac_dyn_a @ state +
                jac_dyn_b @ action +
                jac_dyn_e @ noise),
            out_constant=dynamical_sys.out_eq - (
                jac_out_c @ state +
                jac_out_d @ action),
        )

    def _evaluate_sym(self, sym: sympy.Matrix, values: Dict[str, Union[np.ndarray, float]]) -> Union[np.ndarray, float]:
        return np.array(sym.evalf(subs=values), dtype=np.float64)

    def discretization(self,
                       affine_sys: AffineSystem,
                       lin_point: InputValues,
                       method: str = "exact"
                       ) -> AffineDiscreteSystem:
        values = self.fill_symbols(lin_point, affine_sys)
        continouos_matrices = {key: self._evaluate_sym(getattr(affine_sys, f"{key}_matrix"), values)
                               for key in ("a", "b", "c", "e", "d")}
        constants = {f"{prefix}_constant": self._evaluate_sym(getattr(affine_sys, f"{prefix}_constant"), values).flatten()
                     for prefix in ("dyn", "out")}

        if method == "exact":
            method_fn = self._exact_discretization
        elif method == "euler":
            raise NotImplementedError
            method_fn = self._euler_discretization
        else:
            raise ValueError(f"Unknown discretization method: {method}")
        return method_fn(continouos_matrices, constants, dt=values["dt"])

    def _exact_discretization(self,
                              continouos_matrices: Dict[str, np.ndarray],
                              constants: Dict[str, np.ndarray],
                              dt: float,
                              ) -> AffineDiscreteSystem:
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

        return AffineDiscreteSystem(
            a_matrix=discerete_a_matrix,
            b_matrix=discerete_b_matrix,
            e_matrix=discerete_e_matrix,
            c_matrix=continouos_matrices["c"],
            d_matrix=continouos_matrices["d"],
            dyn_constant=(exp_a @ np.diag(constants["dyn_constant"])),
            out_constant=(np.diag(constants["out_constant"]))
        )

    def _euler_discretization(self,
                              continouos_matrices: Dict[str, np.ndarray],
                              dt: float,
                              *args,
                              ) -> AffineDiscreteSystem:
        raise NotImplementedError
        return dict(
            a_matrix=np.eye(continouos_matrices["a"].shape[0]) + continouos_matrices["a"] * dt,
            b_matrix=continouos_matrices["b"] * dt,
            e_matrix=continouos_matrices["e"] * dt,
            c_matrix=continouos_matrices["c"],
            d_matrix=continouos_matrices["d"],
        )

    def augmentation(self,
                     affine_dsc_sys: AffineDiscreteSystem,
                     costs: QuadraticCosts,
                     constraints: LinearConstraints
                     ) -> NominalLinearEnvParams:

        return NominalLinearEnvParams(
            matrices=LinearDiscreteSystem(
                a_matrix=np.block([
                    [affine_dsc_sys.a_matrix, affine_dsc_sys.dyn_constant],
                    [np.zeros_like(affine_dsc_sys.dyn_constant.T), np.eye(self.state_size)]
                ]),
                b_matrix=np.block([
                    [affine_dsc_sys.b_matrix],
                    [np.zeros_like(affine_dsc_sys.b_matrix)]
                ]),
                e_matrix=np.block([
                    [affine_dsc_sys.e_matrix],
                    [np.zeros_like(affine_dsc_sys.e_matrix)]
                ]),
                c_matrix=np.block([
                    [affine_dsc_sys.c_matrix, affine_dsc_sys.out_constant],
                    [np.zeros_like(affine_dsc_sys.out_constant.T), np.eye(self.state_size)]
                ]),
                d_matrix=np.block([
                    [affine_dsc_sys.d_matrix],
                    [np.zeros_like(affine_dsc_sys.d_matrix)]
                ])
            ),
            costs=QuadraticCosts(
                state=np.block([
                    [costs.state, np.zeros_like(costs.state)],
                    [np.zeros_like(costs.state), np.zeros_like(costs.state)]
                ]),
                action=costs.action,
                final=np.block([
                    [costs.final, np.zeros_like(costs.final)],
                    [np.zeros_like(costs.final), np.zeros_like(costs.final)]
                ]),
            ),
            constraints=LinearConstraints(
                state_constraint_matrix=np.block([
                    [constraints.state_constraint_matrix, np.zeros_like(
                        constraints.state_constraint_matrix)]
                ]),
                state_constraint_vector=constraints.state_constraint_vector,
                action_constraint_matrix=constraints.action_constraint_matrix,
                action_constraint_vector=constraints.action_constraint_vector,
            )
        )


class LinearizationWrapper(gym.ObservationWrapper):

    def __init__(self, env: Plant):
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError("Plant must have Box observation shape!")
        if len(env.observation_space.shape) != 1:
            raise ValueError("Plant must be 1 dimensinoal shape!")
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            shape=(env.state_size * 2,),
            low=np.concatenate([env.observation_space.low,
                               np.ones_like(env.observation_space.low)]),
            high=np.concatenate([env.observation_space.high,
                                np.ones_like(env.observation_space.high)])
        )

        self.output_disturbance = np.concatenate([
            env.output_disturbance, np.zeros_like(env.output_disturbance)
        ], axis=0)
        self.reference_sequence = np.concatenate([
            env.reference_sequence, np.ones_like(env.reference_sequence)
        ], axis=0)
        self.state_disturbance = env.state_disturbance
        self.action_disturbance = env.action_disturbance

        self.state_size = env.state_size * 2
        self.action_size = env.action_size
        self.noise_size = env.noise_size
        self.output_size = env.output_size * 2

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return np.concatenate([obs, np.ones_like(obs)])

    def nominal_model(self,
                      lin_point: InputValues,
                      discretization_method: str = "exact"
                      ) -> NominalLinearEnvParams:
        return self.env.linearization(
            lin_point=lin_point,
            discretization_method=discretization_method)
