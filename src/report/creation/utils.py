import copy
from typing import Any

import numpy as np
import pandas as pd
from orion import Orion

from src.constants.custom_types import CLASSIFICATION_REPORT_TYPE, METRIC_REPOSITORY_TYPE
from src.report.creation.constants import WORDS_IN_DESCRIPTION_PER_ROW


def reformat_description(description: str) -> list[list[str]]:
    split_description = description.split()
    reformatted_description = []

    start = 0
    while len(reformatted_description) * WORDS_IN_DESCRIPTION_PER_ROW <= len(split_description):
        end = min(start + WORDS_IN_DESCRIPTION_PER_ROW, len(split_description))
        reformatted_description.append([' '.join(split_description[start: end])])
        start += WORDS_IN_DESCRIPTION_PER_ROW

    return reformatted_description

# todo: перенести так как используется не только тут
def get_window_size_rolling_window_sequences(orion_model: Orion) -> int:
    if 'mlstars.custom.timeseries_preprocessing.rolling_window_sequences#1' in orion_model._hyperparameters.keys():
        hyperparameters = orion_model._hyperparameters[
            'mlstars.custom.timeseries_preprocessing.rolling_window_sequences#1']
        if 'window_size' in hyperparameters.keys():
            return int(hyperparameters['window_size'])

    return int(
        orion_model._mlpipeline.init_params['mlstars.custom.timeseries_preprocessing.rolling_window_sequences#1'][
            'window_size'])


def get_window_size_find_anomalies(orion_model: Orion) -> int:
    if 'custom.find_anomalies#1' in orion_model._hyperparameters.keys():
        hyperparameters = orion_model._hyperparameters['custom.find_anomalies#1']
        if 'window_size' in hyperparameters.keys():
            return int(hyperparameters['window_size'])

    return 2000  # дефолтное значение, зашитое внтури библиотеки


def get_clustering_flag(orion_model: Orion) -> bool:
    if 'custom.find_anomalies#1' in orion_model._hyperparameters.keys():
        hyperparameters = orion_model._hyperparameters['custom.find_anomalies#1']
        if 'clustering' in hyperparameters.keys():
            return hyperparameters['clustering']

    return orion_model._mlpipeline.init_params['custom.find_anomalies#1']['clustering']


def get_orion_processing_params(orion_model: Orion) -> dict[str, Any]:
    orion_processing_params = {
        'window_size_rolling_window_sequences': get_window_size_rolling_window_sequences(orion_model),
        'clustering': get_clustering_flag(orion_model)
    }

    if orion_processing_params['clustering'] is False:
        orion_processing_params['window_size_find_anomalies'] = get_window_size_find_anomalies(orion_model)

    return orion_processing_params


def get_df_from_classification_report(mean_classification_report: CLASSIFICATION_REPORT_TYPE) -> pd.DataFrame:
    report = copy.deepcopy(mean_classification_report)

    x1, x2 = report['accuracy']
    report['accuracy'] = {'f1-score': x1,
                          'support': x2}

    return pd.DataFrame(report).transpose()


def get_df_from_list(list_values: list[Any]) -> pd.DataFrame:
    return pd.DataFrame(list_values, index=None)


def save(report_file_name: str,
         df: pd.DataFrame, mode: str,
         index: bool = False,
         header: bool = False,
         blank_before: bool = False,
         blank_after: bool = False) -> None:
    empty_df = pd.DataFrame([[np.nan] * len(df.columns)], columns=df.columns)

    if blank_before and not blank_after:
        result_df = pd.concat([empty_df, df], ignore_index=True)
    elif not blank_before and blank_after:
        result_df = pd.concat([df, empty_df], ignore_index=True)
    elif blank_before and blank_after:
        result_df = pd.concat([empty_df, df, empty_df], ignore_index=True)
    else:
        result_df = df

    result_df.to_csv(report_file_name, mode=mode, index=index, header=header)


def get_metrics_from_classification_report(mean_classification_report: CLASSIFICATION_REPORT_TYPE) -> dict[str, float]:
    metrics = dict()

    for key, value in mean_classification_report.items():
        if key in ['accuracy', 'macro avg', 'weighted avg']:
            if isinstance(value, list):
                metrics['accuracy'] = value[0]
            else:
                for name, metric_value in value.items():
                    if name != 'support':
                        full_metric_name = '_'.join(key.split()) + '_' + name.replace('-', '_')
                        metrics[full_metric_name] = metric_value

    return metrics


def parse_classification_report(repository: METRIC_REPOSITORY_TYPE) -> METRIC_REPOSITORY_TYPE:
    for model_name, model_metrics in repository.items():
        for metric_id, metric_value in model_metrics.items():
            if metric_id == 'classification_report':
                metrics_from_classification_report = get_metrics_from_classification_report(metric_value)
                repository[model_name].update(metrics_from_classification_report)
                repository[model_name].pop('classification_report')
                break

    return repository
