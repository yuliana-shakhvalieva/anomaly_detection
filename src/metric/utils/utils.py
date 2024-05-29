from typing import Any

import numpy as np
import pandas as pd

from src.common import constants
from src.constants.custom_types import CLASSIFICATION_REPORT_TYPE


def get_mean_classification_report(
        classification_reports: list[CLASSIFICATION_REPORT_TYPE]) -> CLASSIFICATION_REPORT_TYPE:
    mean_classification_report = classification_reports[0]

    for key, value in mean_classification_report.items():
        if isinstance(value, dict):
            for sub_key in value.keys():
                values = [el[key][sub_key] for el in classification_reports]
                mean_classification_report[key][sub_key] = get_mean_list(values)
        else:
            values = []
            for el in classification_reports:
                if isinstance(el[key], list):
                    values.append(el[key][0])
                else:
                    values.append(el[key])

            mean_classification_report[key] = [get_mean_list(values)]

    mean_classification_report['accuracy'].append(mean_classification_report['weighted avg']['support'])

    return mean_classification_report


def get_mean_list(value: list[float or int]) -> float:
    return round(np.mean(value), 2)


def get_mean(list_values: list[float] or list[CLASSIFICATION_REPORT_TYPE]) -> float or CLASSIFICATION_REPORT_TYPE:
    if len(list_values) == 0:
        return list_values

    if isinstance(list_values[0], dict):
        return get_mean_classification_report(list_values)
    return get_mean_list(list_values)


def single_anomaly_check(df: pd.Series, index: list[int]) -> bool:
    if df.iloc[index[0]] == 1:
        return True
    else:
        return False


def range_anomaly_check(df: pd.Series, index: list[int]) -> bool:
    start, end = index
    count_detected_anomalies = np.sum(df.iloc[start: end])

    if count_detected_anomalies > 0:
        return True
    else:
        return False


def anomaly_detected(df: pd.Series, index: int or list[int]) -> bool:
    if isinstance(index, int):
        index = [index]

    if len(index) == 1:
        return single_anomaly_check(df, index)
    elif len(index) == 2:
        return range_anomaly_check(df, index)
    else:
        raise ValueError


def get_perfect_range(i: int, total: int) -> (int, int):
    start = max(0, i - constants.EPS)
    end = min(i + constants.EPS + 1, total)

    return start, end


def flatten(input_list: list[list[Any]]) -> list[Any]:
    return list(np.concatenate(input_list).flat)
