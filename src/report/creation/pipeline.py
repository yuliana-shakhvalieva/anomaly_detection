import copy
from typing import Any

import pandas as pd

import src.report.creation.constants as constants
from src.constants.custom_types import METRIC_REPOSITORY_TYPE
from src.metric.constants.descriptions import METRICS_DESCRIPTIONS
from src.metric.constants.names import CLASSIFICATION_REPORT
from src.model.proxy import OrionModelProxy
from src.report.common.utils import get_df_with_best_keys_by_metrics
from src.report.creation import utils
from src.report.creation.utils import parse_classification_report


def start_report(report_file_name: str, description: str) -> None:
    write_values = [[constants.TITLE]]

    description = utils.reformat_description(description)
    write_values.extend(description)

    utils.save(report_file_name, utils.get_df_from_list(write_values), 'w', blank_after=True, blank_before=True)


def add_data_generation_params(report_file_name: str) -> None:
    utils.save(report_file_name, utils.get_df_from_list(constants.TITLE_DATA_GENERATION_PARAMS), 'a')
    utils.save(report_file_name, utils.get_df_from_list(constants.DATA_GENERATION_PARAMS), 'a', blank_after=True)


def add_anomaly_generation_params(report_file_name: str, anomaly_generation_params: dict[str, Any]) -> None:
    utils.save(report_file_name, utils.get_df_from_list(constants.TITLE_ANOMALY_GENERATION_PARAMS), 'a')

    write_anomaly_generation_params = copy.deepcopy(constants.ANOMALY_GENERATION_PARAMS)

    for args in write_anomaly_generation_params:
        _, arg_name = args
        arg_value = anomaly_generation_params[arg_name]
        args.append(arg_value)

    utils.save(report_file_name, utils.get_df_from_list(write_anomaly_generation_params), 'a', blank_after=True)


def add_other_general_params(report_file_name: str) -> None:
    utils.save(report_file_name, utils.get_df_from_list(constants.TITLE_OTHER_GENERAL_PARAMS), 'a')
    utils.save(report_file_name, utils.get_df_from_list(constants.OTHER_GENERAL_PARAMS), 'a', blank_after=True)


def add_model_results(report_file_name: str, model: OrionModelProxy, mean_metrics: dict) -> None:
    utils.save(report_file_name, utils.get_df_from_list([model.name]), 'a', blank_before=True)

    utils.save(report_file_name,
               utils.get_df_from_classification_report(mean_metrics[CLASSIFICATION_REPORT]),
               'a',
               index=True,
               header=True)

    write_metrics = [[METRICS_DESCRIPTIONS[metric_id], mean_value]
                     for metric_id, mean_value in mean_metrics.items() if metric_id != CLASSIFICATION_REPORT]
    utils.save(report_file_name, utils.get_df_from_list(write_metrics), 'a', blank_before=True)


def add_model_params(report_file_name: str, model: OrionModelProxy) -> None:
    write_params = [[param_name, param_value] for param_name, param_value in model.model_hyperparameters.items()]
    utils.save(report_file_name, utils.get_df_from_list(write_params), 'a', blank_before=True)


def add_orion_processing_params(report_file_name: str, model: OrionModelProxy) -> None:
    write_params = [[param_name, param_value] for param_name, param_value in
                    utils.get_orion_processing_params(model.model).items()]

    utils.save(report_file_name, utils.get_df_from_list(write_params), 'a', blank_before=True, blank_after=True)


def add_best_models_by_metrics(report_file_name: str,
                               metric_repository: METRIC_REPOSITORY_TYPE) -> (METRIC_REPOSITORY_TYPE, pd.DataFrame):
    metrics_by_models = parse_classification_report(metric_repository)
    df_best_models_names = get_df_with_best_keys_by_metrics(metrics_by_models,
                                                            pd.DataFrame(
                                                                columns=['metric', 'best_model', 'best_value']))

    utils.save(report_file_name,
               utils.get_df_from_list(constants.TITLE_BEST_MODELS_BY_METRICS), 'a', blank_before=True)
    utils.save(report_file_name, df_best_models_names, 'a')

    return metrics_by_models, df_best_models_names


def get_best_final_model(df: pd.DataFrame) -> list[str]:
    best_models = df.best_model.explode()
    return best_models.mode()
