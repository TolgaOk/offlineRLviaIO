from typing import List, Dict, Union, Any, Callable, Tuple, Type
from functools import partial
import numpy as np

from io_agent.plant.base import Plant, LinearizationWrapper, InputValues
from io_agent.evaluator import ControlLoop, Transition
from io_agent.control.mpc import MPC
from io_agent.control.rmpc import RobustMPC
from io_agent.control.io import IOController, AugmentedTransition, AugmentDataset
from io_agent.utils import FeatureHandler


def run_agent(agent: Union[MPC, IOController],
              plant: Plant,
              use_foresight: bool,
              bias_aware: bool,
              env_reset_rng: np.random.Generator = None,
              ) -> List[Transition]:
    """ Simulate the agent in the Fighter environment for 1 trajectory

    Args:
        agent (MPC): MPC or IO controller
        env_length (int): Length of the environment
        use_foresight (bool): If true, feed the agent with future noise signal
        disturbance_bias (Optional[np.ndarray], optional): Bias for the state disturbance. Defaults to 0.
        bias_aware (bool, optional): _description_. If true, feed the agent with actual noise (biased).

    Returns:
        List[Transition]: Trajectory of transitions
    """
    evaluator = ControlLoop(
        plant=plant,
        controller=agent,
        rng=env_reset_rng
    )
    return evaluator.simulate(
        bias_aware=bias_aware,
        use_foresight=use_foresight,
    )


def run_mpc(plant: Plant,
            horizon: int = 20,
            use_foresight: bool = True,
            bias_aware: bool = True,
            env_reset_rng: np.random.Generator = None,
            ) -> List[Transition]:
    """ Run MPC agent

    Args:
        horizon (int, optional): Noise horizon of MPC. Defaults to 20.
        env_length (int): Length of the environment
        use_foresight (bool): If true, feed the agent with future noise signal
        disturbance_bias (Optional[np.ndarray], optional): Bias for the state disturbance. Defaults to 0.
        bias_aware (bool, optional): _description_. If true, feed the agent with actual noise (biased).

    Returns:
        List[Transition]: Trajectory of transitions
    """
    plant = LinearizationWrapper(plant)
    agent = MPC(
        action_size=plant.action_size,
        state_size=plant.state_size,
        noise_size=plant.noise_size,
        output_size=plant.output_size,
        horizon=horizon)
    agent.optimizer = agent.prepare_optimizer(
        plant.nominal_model(
            lin_point=None
        ))
    return run_agent(
        agent=agent,
        plant=plant,
        use_foresight=use_foresight,
        bias_aware=bias_aware,
        env_reset_rng=env_reset_rng,
    )


def run_rmpc(plant: Plant,
             horizon: int = 20,
             use_foresight: bool = True,
             rho: float = 0.1,
             bias_aware: bool = True,
             env_reset_rng: np.random.Generator = None,
             ) -> List[Transition]:
    """ Run Robust MPC

    Args:
        horizon (int, optional): Noise horizon of MPC. Defaults to 20.
        rho (float, optional): Robustness radius. Defaults to 0.1.
        env_length (int): Length of the environment
        use_foresight (bool): If true, feed the agent with future noise signal
        disturbance_bias (Optional[np.ndarray], optional): Bias for the state disturbance. Defaults to 0.
        bias_aware (bool, optional): _description_. If true, feed the agent with actual noise (biased).

    Returns:
        List[Transition]: Trajectory of transitions
    """
    plant = LinearizationWrapper(plant)
    agent = RobustMPC(action_size=plant.action_size,
                      state_size=plant.state_size,
                      noise_size=plant.noise_size,
                      output_size=plant.output_size,
                      horizon=horizon,
                      rho=rho,
                      state_constraints_flag=True,
                      input_constraints_flag=True)
    agent.optimizer = agent.prepare_optimizer(
        plant.nominal_model(
            lin_point=None
        ))
    return run_agent(
        agent=agent,
        plant=plant,
        use_foresight=use_foresight,
        bias_aware=bias_aware,
        env_reset_rng=env_reset_rng,
    )


def prepare_io(dataset: List[Transition],
               plant: Plant,
               expert_class: Union[Type[MPC], Type[RobustMPC]],
               expert_kwargs: Dict[str, Any],
               n_past: int = 1,
               add_bias: bool = True,
               ) -> Tuple[Union[
                   Plant,
                   FeatureHandler,
                   List[AugmentedTransition]]]:
    plant = LinearizationWrapper(plant)
    nominal_model = plant.nominal_model(
        lin_point=None
    )
    feature_handler = FeatureHandler(
        params=nominal_model,
        n_past=n_past,
        add_bias=add_bias,
        use_action_regressor=False,
        use_noise_regressor=True,
        use_state_regressor=False)
    expert_agent = expert_class(
        action_size=plant.action_size,
        state_size=plant.state_size,
        noise_size=plant.noise_size,
        output_size=plant.output_size,
        **expert_kwargs)
    expert_agent.optimizer = expert_agent.prepare_optimizer(nominal_model)
    augmenter = AugmentDataset(
        expert_agent=expert_agent,
        feature_handler=feature_handler
    )
    augmented_dataset = augmenter(dataset)
    return plant, feature_handler, augmented_dataset


def run_io(feature_handler: FeatureHandler,
           plant: Plant,
           augmented_dataset: List[AugmentedTransition],
           dataset_permute_rng: np.random.Generator,
           dataset_length: int = 300,
           bias_aware: bool = False,
           ) -> Callable[[Any], Any]:

    io_agent = IOController(
        params=feature_handler.params,
        include_constraints=True,
        soften_state_constraints=True,
        state_constraints_flag=True,
        action_constraints_flag=True,
        dataset_length=dataset_length,
        feature_handler=feature_handler)
    io_agent.train(
        augmented_dataset,
        rng=dataset_permute_rng)
    io_agent.action_optimizer = io_agent.prepare_action_optimizer()
    return partial(run_agent,
                   agent=io_agent,
                   plant=plant,
                   bias_aware=bias_aware,
                   use_foresight=False,   # IO agent does not look into the future
                   )


def run_io_mpc(dataset: List[Transition],
               plant: Plant,
               dataset_permute_rng: np.random.Generator,
               n_past: int = 1,
               add_bias: bool = True,
               dataset_length: int = 300,
               bias_aware: bool = True,
               expert_horizon: int = 20,
               ) -> Callable[[Any], Any]:
    """ Train and simulate IO agent with MPC as the expert

    Args:
        dataset (List[Transition]): List if transitions to be used
            as the training data
        horizon (int, optional): Horizon of the expert agent. Defaults to 20.
        env_length (int, optional): Length of the environment. Defaults to 60.
        disturbance_bias (Optional[np.ndarray], optional): Bias for the state disturbance. Defaults to 0.
        bias_aware (bool, optional): _description_. If true, feed the agent with actual noise (biased).

    Returns:
        List[Transition]: Trajectory of transitions
    """
    (linearized_plant,
     feature_handler,
     augmented_dataset
     ) = prepare_io(
        dataset=dataset,
        plant=plant,
        expert_class=MPC,
        expert_kwargs={"horizon": expert_horizon},
        n_past=n_past,
        add_bias=add_bias,
    )

    return run_io(
        plant=linearized_plant,
        feature_handler=feature_handler,
        augmented_dataset=augmented_dataset,
        dataset_permute_rng=dataset_permute_rng,
        dataset_length=dataset_length,
        bias_aware=bias_aware,
    )


def run_io_rmpc(dataset: List[Transition],
                plant: Plant,
                dataset_permute_rng: np.random.Generator,
                expert_rho: float,
                n_past: int = 1,
                add_bias: bool = True,
                dataset_length: int = 300,
                bias_aware: bool = True,
                expert_horizon: int = 20,
                ) -> Callable[[Any], Any]:
    """ Train and simulate IO agent with Robust MPC as the expert

    Args:
        dataset (List[Transition]): List if transitions to be used
            as the training data
        horizon (int, optional): Horizon of the expert agent. Defaults to 20.
        env_length (int, optional): Length of the environment. Defaults to 60.
        disturbance_bias (Optional[np.ndarray], optional): Bias for the state disturbance. Defaults to 0.
        bias_aware (bool, optional): _description_. If true, feed the agent with actual noise (biased).

    Returns:
        List[Transition]: Trajectory of transitions
    """
    (linearized_plant,
     feature_handler,
     augmented_dataset
     ) = prepare_io(
        dataset=dataset,
        plant=plant,
        expert_class=RobustMPC,
        expert_kwargs={
            "horizon": expert_horizon,
            "rho": expert_rho,
            "state_constraints_flag": True,
            "input_constraints_flag": True
        },
        n_past=n_past,
        add_bias=add_bias,
    )
    return run_io(
        plant=linearized_plant,
        feature_handler=feature_handler,
        augmented_dataset=augmented_dataset,
        dataset_permute_rng=dataset_permute_rng,
        dataset_length=dataset_length,
        bias_aware=bias_aware,
    )
