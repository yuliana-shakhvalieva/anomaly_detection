import os
import shutil
import time
from typing import Callable, Any

from src.constants.custom_types import ORION_HYPERPARAMETERS_TYPE


def measure_seconds(func: Callable, *args, **kwargs) -> (Any, int):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    spent_time = end - start

    return result, spent_time


def concat_lists(list1: list[Any], list2: list[Any]) -> list[Any]:
    return list1 + list2


def prepare_path(paths: str or list[str]) -> None:
    if isinstance(paths, str):
        paths = [paths]

    for path in paths:
        os.makedirs(path, exist_ok=True)

        for obj in os.scandir(path):
            if obj.is_file():
                os.chmod(obj, 0o777)
                os.remove(obj)
            else:
                shutil.rmtree(obj)


def get_inner_dict(dictionary: dict, keys: list) -> dict:
    if len(keys) == 0:
        return dictionary

    local_dictionary = get_inner_dict(dictionary, keys[:-1])

    if keys[-1] not in local_dictionary:
        local_dictionary[keys[-1]] = {}

    return local_dictionary[keys[-1]]


def add_to_dict_list(dictionary: dict, keys: str or list[str], value: Any) -> None:
    if isinstance(keys, str):
        keys = [keys]

    inner_dict = get_inner_dict(dictionary, keys[:-1])
    if keys[-1] not in inner_dict:
        inner_dict[keys[-1]] = [value]
    else:
        inner_dict[keys[-1]].append(value)


def concat_orion_hyperparameters_dict(minor_priority: ORION_HYPERPARAMETERS_TYPE,
                                      major_priority: ORION_HYPERPARAMETERS_TYPE) -> ORION_HYPERPARAMETERS_TYPE:
    for key, value in major_priority.items():
        for k, v in value.items():
            if key not in minor_priority.keys():
                minor_priority[key] = dict()
            minor_priority[key][k] = v

    return minor_priority
