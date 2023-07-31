from typing import List, Optional, Dict, Any, Callable
from itertools import chain
from functools import partial
import os
import pickle
from multiprocessing import Process, Queue
from tqdm.notebook import tqdm
import traceback
import queue
from warnings import warn
import cvxpy as cp

from io_agent.evaluator import Transition


def steady_state_cost(trajectories: List[List[Transition]], ratio: float) -> List[float]:
    return [transition.cost for transition in chain(
        *[trajectory[int(len(trajectory) * (1 - ratio)):] for trajectory in trajectories]
    )]


def parallelize(n_proc: int,
                fn: Callable[[Any], Any],
                kwargs_list: List[Dict[str, Any]],
                loading_bar_kwargs: Optional[Dict[str, Any]] = None
                ) -> List[Any]:

    def _async_execute_wrapper() -> None:
        while True:
            try:
                kwargs, key = work_queue.get(block=False)
                result = fn(**kwargs)
                result_queue.put({"key": key, "result": result})
            except queue.Empty:
                if work_queue.qsize() == 0:
                    return None
            except Exception as err:
                result_queue.put({"key": key, "result": None})
                traceback.print_exc()

    result_queue = Queue()
    work_queue = Queue()

    for key, kwargs in enumerate(kwargs_list):
        work_queue.put((kwargs, key))

    loading_bar = (partial(tqdm, **loading_bar_kwargs)
                   if loading_bar_kwargs is not None
                   else lambda x: x)

    process_list = []
    for _ in range(n_proc):
        process_list.append(Process(target=_async_execute_wrapper))
        process_list[-1].start()

    results_dict = {}
    for _ in loading_bar(range(len(kwargs_list))):
        _return = result_queue.get(block=True)
        results_dict[_return["key"]] = _return["result"]

    for process in (process_list):
        process.join()

    results = [results_dict[index] for index in range(len(results_dict))]
    return results


def save_experiment(values: Any, seed: int, exp_dir: str, name: str) -> None:
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, f"{name}-{seed}"), "wb") as fobj:
        pickle.dump(values, fobj)


def load_experiment(path: str) -> Any:
    with open(os.path.join(path), "rb") as fobj:
        return pickle.load(fobj)


def try_solve(patience: int, verbose: bool = True):
    def decorator(function: Callable[[Any], Any]) -> Callable[[Any], Any]:
        def wrapper(*args, **kwargs) -> Any:
            for attempt in range(1, patience + 1):
                try:
                    return function(*args, **kwargs)
                except cp.SolverError as err:
                    if verbose:
                        print(f"Failed to solve at attempt: {attempt}")
            raise err
        return wrapper
    return decorator
