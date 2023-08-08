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
class LinearConstraint:
    matrix: np.ndarray
    vector: np.ndarray


@dataclass
class LinearConstraints:
    state: Optional[LinearConstraint] = None
    action: Optional[LinearConstraint] = None


@dataclass
class Disturbances:
    state: Optional[np.ndarray] = None
    action: Optional[np.ndarray] = None
    output: Optional[np.ndarray] = None


@dataclass
class InputValues():
    state: np.ndarray
    action: np.ndarray
    noise: np.ndarray
    output_noise: Optional[np.ndarray] = None


@dataclass
class SystemInput():
    state: sympy.Matrix
    action: sympy.Matrix
    noise: sympy.Matrix
    output_noise: Optional[sympy.Matrix] = None


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

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 noise_size: int,
                 output_size: int,
                 max_length: int,
                 costs: QuadraticCosts,
                 constraints: LinearConstraints,
                 reference_sequence: Optional[np.ndarray] = None,
                 disturbance_bias: Optional[Disturbances] = None,
                 ) -> None:
        """ Base Plant class with quadratic costs and optinal linear constraints and reference_sequence
        """
        self.state_size = state_size
        self.action_size = action_size
        self.noise_size = noise_size
        self.output_size = output_size
        self.max_length = max_length
        self.costs = costs
        self.constraints = constraints
        self.disturbance_bias = disturbance_bias

        if reference_sequence is None:
            reference_sequence = np.zeros((self.output_size, self.max_length * 2))
        self.reference_sequence = reference_sequence
        super().__init__()

    @abstractmethod
    def generate_disturbance(self, rng: np.random.Generator) -> Disturbances:
        raise NotImplementedError

    @abstractmethod
    def symbolic_dynamical_system(self) -> DynamicalSystem:
        raise NotImplementedError

    @abstractmethod
    def fill_symbols(self,
                     input_values: InputValues,
                     ) -> Dict[str, Union[float, np.ndarray]]:
        raise NotImplementedError

    @abstractmethod
    def _reset(self,
               rng: np.random.Generator,
               options: Optional[Dict[str, Any]] = None
               ) -> Tuple[Union[np.ndarray, Optional[Dict[str, Disturbances]]]]:
        raise NotImplementedError

    @abstractmethod
    def default_lin_point(self) -> Optional[InputValues]:
        raise NotImplemented

    def reset_disturbances(self,
                           rng: np.random.Generator,
                           ) -> Tuple[Disturbances]:
        env_disturbance = dict()
        biased_disturbance = dict()
        for key, dist in asdict(self.generate_disturbance(rng)).items():
            if dist is None:
                env_disturbance[key] = np.zeros((getattr(self, f"{key}_size"),
                                                 self.max_length * 2))
            else:
                env_disturbance[key] = dist.copy()
            biased_disturbance[key] = env_disturbance[key].copy()
        if (self.disturbance_bias is not None):
            for key, dist_bias in asdict(self.disturbance_bias).items():
                if (dist_bias is not None):
                    env_disturbance[key] += dist_bias
        return Disturbances(**env_disturbance), Disturbances(**biased_disturbance)

    def reset(self,
              seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None
              ) -> Tuple[Union[np.ndarray, Optional[Dict[str, Disturbances]]]]:
        """ Initialize the environment with random initial state

        Args:
            seed (Optional[int], optional): Random seed of the episode/trajectory. Defaults to random integer.
            options (Optional[Dict[str, Any]], optional): Options for the episode (unused). Defaults to None.

        Returns:
            Tuple[Union[np.ndarray, float, bool, Optional[Dict[str, Any]]]]:
                - initial output/state (np.ndarray): Array of shape (S,)
                     where S denotes the state/input size
                - info (Dict[str, Any]): Metadata of the initial state (set to None)
        """
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        if options is None:
            bias_aware = False
        else:
            bias_aware = options.get("bias_aware")
        self.disturbances, biased_disturbances = self.reset_disturbances(rng)
        return_dist = self.disturbances if bias_aware else biased_disturbances

        output, info = self._reset(rng, options)
        if "disturbance" in info.keys():
            raise RuntimeError("The name `disturbance` is used by base Plant in the info!")

        return output, dict(**info, disturbance=return_dist)

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

    def evaluate_sym(self, sym: sympy.Matrix, values: Dict[str, Union[np.ndarray, float]]) -> Union[np.ndarray, float]:
        return np.array(sym.evalf(subs=values), dtype=np.float64)

    def discretization(self,
                       affine_sys: AffineSystem,
                       lin_point: InputValues,
                       method: str = "exact"
                       ) -> AffineDiscreteSystem:
        values = self.fill_symbols(lin_point)
        continuous_matrices = {key: self.evaluate_sym(getattr(affine_sys, f"{key}_matrix"), values)
                               for key in ("a", "b", "c", "e", "d")}
        constants = {f"{prefix}_constant": self.evaluate_sym(getattr(affine_sys, f"{prefix}_constant"), values).flatten()
                     for prefix in ("dyn", "out")}

        if method == "exact":
            method_fn = self._exact_discretization
        elif method == "euler":
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown discretization method: {method}")
        return method_fn(continuous_matrices, constants, dt=values["dt"])

    def _exact_discretization(self,
                              continuous_matrices: Dict[str, np.ndarray],
                              constants: Dict[str, np.ndarray],
                              dt: float,
                              ) -> AffineDiscreteSystem:
        n_state = continuous_matrices["a"].shape[0]
        n_action = continuous_matrices["b"].shape[1]
        n_noise = continuous_matrices["e"].shape[1]
        n_out = continuous_matrices["c"].shape[0]

        n_size = n_state * 2 + n_action + n_noise
        matrix = np.zeros((n_size, n_size))
        matrix[:n_state, :n_state] = continuous_matrices["a"]
        matrix[:n_state, n_state: 2 * n_state] = np.eye(n_state)
        matrix[:n_state, 2 * n_state:-n_noise] = continuous_matrices["b"]
        matrix[:n_state, 2 * n_state + n_action:] = continuous_matrices["e"]

        exp_matrix = scipy.linalg.expm(matrix * dt)
        discerete_a_matrix = exp_matrix[:n_state, :n_state]
        discerete_b_matrix = exp_matrix[:n_state, 2 * n_state:-n_noise]
        exp_a = exp_matrix[:n_state, n_state: 2 * n_state]
        discerete_e_matrix = exp_matrix[:n_state, 2 * n_state + n_action:]

        out_constant = np.zeros_like(continuous_matrices["c"])
        out_constant[:n_out, :n_out] = np.diag(constants["out_constant"])

        return AffineDiscreteSystem(
            a_matrix=discerete_a_matrix,
            b_matrix=discerete_b_matrix,
            e_matrix=discerete_e_matrix,
            c_matrix=continuous_matrices["c"],
            d_matrix=continuous_matrices["d"],
            dyn_constant=(exp_a @ np.diag(constants["dyn_constant"])),
            out_constant=out_constant
        )

    def _euler_discretization(self,
                              continuous_matrices: Dict[str, np.ndarray],
                              dt: float,
                              *args,
                              ) -> AffineDiscreteSystem:
        raise NotImplementedError
        return dict(
            a_matrix=np.eye(continuous_matrices["a"].shape[0]) + continuous_matrices["a"] * dt,
            b_matrix=continuous_matrices["b"] * dt,
            e_matrix=continuous_matrices["e"] * dt,
            c_matrix=continuous_matrices["c"],
            d_matrix=continuous_matrices["d"],
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
                ]),
                d_matrix=np.block([
                    [affine_dsc_sys.d_matrix],
                ])
            ),
            costs=costs,
            constraints=LinearConstraints(
                state=LinearConstraint(
                    matrix=np.block([
                        [constraints.state.matrix, np.zeros_like(constraints.state.matrix)]
                    ]),
                    vector=constraints.state.vector
                ) if constraints.state is not None else None,
                action=constraints.action
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

        # self.disturbances = env.disturbances
        self.reference_sequence = env.reference_sequence

        self.state_size = env.state_size * 2
        self.action_size = env.action_size
        self.noise_size = env.noise_size
        self.output_size = env.output_size

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return np.concatenate([obs, np.ones_like(obs)])

    def nominal_model(self,
                      lin_point: Optional[InputValues] = None,
                      discretization_method: str = "exact"
                      ) -> NominalLinearEnvParams:
        if lin_point is None:
            lin_point = self.env.default_lin_point()
        return self.env.linearization(
            lin_point=lin_point,
            discretization_method=discretization_method)


class OldSchoolWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, score_range: Optional[Tuple[float]] = None):
        super().__init__(env)
        self.score_range = score_range

    def reset(self, *args, **kwargs) -> np.ndarray:
        state, info = self.env.reset(*args, **kwargs)
        return state

    def step(self, action: np.ndarray, *args, **kwargs) -> Tuple[Union[np.ndarray, float, bool, Dict[str, Any]]]:
        _action = self.env.action_space.low + \
            (action + 1) / 2 * (self.env.action_space.high - self.env.action_space.low)
        next_state, reward, termination, truncation, info = self.env.step(_action, *args, **kwargs)
        return next_state, reward, termination or truncation, info

    def get_normalized_score(self, score: float) -> float:
        if self.score_range is not None:
            min_score, max_score = self.score_range
            return (score - min_score) / (max_score - min_score)
        return score
