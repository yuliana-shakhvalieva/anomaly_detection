import ast
import json
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.common.utils import add_to_dict_list, prepare_path
from src.constants import generation
from src.constants.experiments import START_EXPERIMENTS_DESCRIPTIONS, END_EXPERIMENT_DESCRIPTION, \
    START_EXPERIMENTS_DESCRIPTIONS_BENCH, END_EXPERIMENTS_DESCRIPTIONS_BENCH
from src.constants.experiments import WINDOW_SIZE
from src.report.analysis.utils import get_columns_result_df
from src.utils.experiments.command_line import get_main_path


@dataclass
class File:
    name: str
    path: str


def get_data_files(*, bench: bool = False,
                   directory: str = 'data',
                   custom_file_names: str or list[str] or None = None) -> list[os.DirEntry or File]:

    method = 'is_file' if not bench else 'is_dir'

    if custom_file_names is None:
        return [filename for filename in os.scandir(directory) if getattr(filename, method)()]

    if isinstance(custom_file_names, str):
        custom_file_names = [custom_file_names]

    return [File(name=f'{custom_file_name}.csv', path=f'{directory}/{custom_file_name}')
            for custom_file_name in custom_file_names]


def get_data_from_json(file_name: str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    file = open(file_name)
    data = json.load(file)

    df_test = pd.DataFrame.from_dict(data['df_test'])
    df_train = pd.DataFrame.from_dict(data['df_train'])
    true_anomalies = pd.DataFrame.from_dict(data['true_anomalies'])

    return df_train, df_test, true_anomalies


def get_experiment_path(save_pdf: bool, data_path: str, experiment_params: str) -> str:
    if save_pdf:
        path = data_path + f'{experiment_params}/'
    else:
        path = data_path

    prepare_path(path)
    return path


def get_anomaly_generation_params(p_anomaly: float = generation.P_ANOMALY,
                                  possible_anomaly_length: list[int] = generation.POSSIBLE_ANOMALY_LENGTH,
                                  start_from: int = generation.START_FROM) -> dict[str, Any]:
    return {'p_anomaly': p_anomaly,
            'possible_anomaly_length': possible_anomaly_length,
            'start_from': start_from}


def get_best_window_size_rolling_window_sequences(file_name: str = 'best_window_sizes.csv') -> dict[str, list[int]]:
    df_result = pd.read_csv('best_params/' + file_name)
    reference_values = dict()
    data_name, best_window_size_name, _ = get_columns_result_df(WINDOW_SIZE)

    for index, row in df_result.iterrows():
        data = row[data_name] + '.csv'
        best_window_sizes = ast.literal_eval(row[best_window_size_name])
        for best_window_size in best_window_sizes:
            add_to_dict_list(reference_values, data, int(best_window_size))

    return reference_values


def get_experiment_path_and_report_file_name(args: list[str],
                                             save_pdf: bool,
                                             data_name: str,
                                             experiment_name: str,
                                             experiment_params: str) -> (str, str):
    if data_name.endswith('.csv'):
        data_name = data_name[:-4]
    data_path = get_main_path(args) + f'{experiment_name}/{data_name}/'
    experiment_path = get_experiment_path(save_pdf, data_path, experiment_params)
    report_file_name = experiment_path + f'{experiment_params}.csv'

    return experiment_path, report_file_name


def get_namings(args: list[str],
                save_pdf: bool,
                data_name: str,
                experiment_name: str,
                experiment_params: str) -> (str, str, str):
    experiment_path, report_file_name = get_experiment_path_and_report_file_name(args,
                                                                                 save_pdf,
                                                                                 data_name,
                                                                                 experiment_name,
                                                                                 experiment_params)
    description = f'{START_EXPERIMENTS_DESCRIPTIONS[experiment_name]} {data_name}, {END_EXPERIMENT_DESCRIPTION}'
    return experiment_path, report_file_name, description


def get_namings_bench(args: list[str],
                      save_pdf: bool,
                      data_name: str,
                      experiment_name: str,
                      experiment_params: str) -> (str, str, str):
    experiment_path, report_file_name = get_experiment_path_and_report_file_name(args,
                                                                                 save_pdf,
                                                                                 data_name,
                                                                                 experiment_name,
                                                                                 experiment_params)
    description = (f'{START_EXPERIMENTS_DESCRIPTIONS_BENCH[experiment_name]} {data_name}, '
                   f'{END_EXPERIMENTS_DESCRIPTIONS_BENCH}')
    return experiment_path, report_file_name, description
