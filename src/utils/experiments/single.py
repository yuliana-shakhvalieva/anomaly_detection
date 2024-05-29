import os
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from src.common.constants import LEN_SEQUENCE
from src.constants.custom_types import CUSTOM_HYPERPARAMETERS_TYPE
from src.constants.experiments import FIT_ON_HISTORY_DATA, FIT_DETECT, SINGLE_FIT
from src.metric.constants import objects
from src.metric.manager import MetricManager
from src.metric.measurers import SingleArimaDetectTimeMeasurer
from src.metric.repository import MetricRepository
from src.model.builder import ModelBuilder
from src.model.manager import ModelManager
from src.report.creation.builder import ReportBuilder
from src.tests.manager import TestsManager
from src.utils.dataframes import apply_orion_format
from src.utils.experiments.pipeline import get_namings, get_data_from_json, get_data_files, get_namings_bench
from src.visualization.repository import PlotsRepository
from src.visualization.visualizers import PDFVisualizer, DefaultPyPlotVisualizer


def init_experiment(args: list[str],
                    experiment_name: str,
                    model_names: str or list[str],
                    measurer: SingleArimaDetectTimeMeasurer,
                    data_name: str,
                    hyperparameters: CUSTOM_HYPERPARAMETERS_TYPE,
                    experiment_params: str,
                    save_pdf: bool,
                    show_plots: bool,
                    benchmark: bool = False) -> (ModelManager,
                                                 TestsManager,
                                                 MetricRepository,
                                                 PlotsRepository,
                                                 str,
                                                 str):
    metric_manager = MetricManager(objects.METRIC_CALCULATORS, measurer)
    metric_repository = MetricRepository()
    if benchmark:
        experiment_path, report_file_name, description = get_namings_bench(args,
                                                                           save_pdf,
                                                                           data_name,
                                                                           experiment_name,
                                                                           experiment_params)
    else:
        experiment_path, report_file_name, description = get_namings(args,
                                                                     save_pdf,
                                                                     data_name,
                                                                     experiment_name,
                                                                     experiment_params)

    visualizers = [DefaultPyPlotVisualizer(show_plots),
                   PDFVisualizer(save_pdf, experiment_path)]

    plots_repository = PlotsRepository(visualizers)

    model_builder = ModelBuilder(model_names=model_names, hyperparameters=hyperparameters)
    model_manager = ModelManager(model_builder, metric_manager)

    test_manager = TestsManager(args, model_manager, metric_repository, plots_repository)

    return model_manager, test_manager, metric_repository, plots_repository, report_file_name, description


def end_experiment(metric_repository: MetricRepository,
                   plots_repository: PlotsRepository,
                   model_manager: ModelManager,
                   report_file_name: str,
                   description: str,
                   anomaly_generation_params: dict[str, Any] or None = None) -> None:
    metric_repository.fix_state()
    plots_repository.visualize()

    report = ReportBuilder(model_manager, metric_repository)
    report.save(report_file_name, description, anomaly_generation_params)


def single_random_experiment(args: list[str],
                             experiment_name: str,
                             model_names: str or list[str],
                             data_name: str,
                             sequences: list[pd.DataFrame],
                             hyperparameters: CUSTOM_HYPERPARAMETERS_TYPE,
                             anomaly_generation_params: dict[str, Any],
                             experiment_params: str,
                             save_pdf: bool,
                             show_plots: bool) -> None:
    (model_manager,
     test_manager,
     metric_repository,
     plots_repository,
     report_file_name,
     description) = init_experiment(args,
                                    experiment_name,
                                    model_names,
                                    objects.SINGLE_ARIMA_DETECT_TIME_MEASURER,
                                    data_name,
                                    hyperparameters,
                                    experiment_params,
                                    save_pdf,
                                    show_plots)

    for i, sequence in tqdm(enumerate(sequences), desc='Sequences', leave=True, total=len(sequences)):
        df_train, df_test = apply_orion_format(df_train=sequence.iloc[:(LEN_SEQUENCE // 2)],
                                               df_test=sequence.iloc[(LEN_SEQUENCE // 2):])

        print(f'\nFile: {data_name}_{i}, {experiment_name}: {experiment_params}')

        model_manager.fit(df_train)
        test_manager.make_tests_on_full_df_test(df_test, anomaly_generation_params)

    end_experiment(metric_repository,
                   plots_repository,
                   model_manager,
                   report_file_name,
                   description,
                   anomaly_generation_params)


def single_benchmarking_experiment_fit_on_history_data(args: list[str],
                                                       experiment_name: str,
                                                       model_names: str or list[str],
                                                       data_name: str,
                                                       directory: os.DirEntry,
                                                       hyperparameters: CUSTOM_HYPERPARAMETERS_TYPE,
                                                       window_size: int,
                                                       experiment_params: str,
                                                       save_pdf: bool,
                                                       show_plots: bool) -> None:
    (model_manager,
     test_manager,
     metric_repository,
     plots_repository,
     report_file_name,
     description) = init_experiment(args,
                                    experiment_name,
                                    model_names,
                                    objects.SINGLE_ARIMA_DETECT_TIME_MEASURER,
                                    data_name,
                                    hyperparameters,
                                    experiment_params,
                                    save_pdf,
                                    show_plots,
                                    True)
    data_files = get_data_files(directory=directory.path)

    for i, file in tqdm(enumerate(data_files), desc='Tests', leave=True, total=len(data_files)):
        print(f'\nFile: {data_name}_{i}, {experiment_name}: {experiment_params}')
        df_train, df_test, true_anomalies = get_data_from_json(file.path)
        test_manager.make_tests(window_size, df_train, df_test, true_anomalies, make_fit=True)

    end_experiment(metric_repository,
                   plots_repository,
                   model_manager,
                   report_file_name,
                   description + FIT_ON_HISTORY_DATA)


def single_benchmarking_experiment_fit_detect(args: list[str],
                                              experiment_name: str,
                                              model_names: str or list[str],
                                              data_name: str,
                                              directory: os.DirEntry,
                                              hyperparameters: CUSTOM_HYPERPARAMETERS_TYPE,
                                              window_size: int,
                                              experiment_params: str,
                                              save_pdf: bool,
                                              show_plots: bool) -> None:
    (model_manager,
     test_manager,
     metric_repository,
     plots_repository,
     report_file_name,
     description) = init_experiment(args,
                                    experiment_name,
                                    model_names,
                                    objects.SINGLE_ARIMA_FIT_DETECT_TIME_MEASURER,
                                    data_name,
                                    hyperparameters,
                                    experiment_params,
                                    save_pdf,
                                    show_plots,
                                    True)

    data_files = get_data_files(directory=directory.path)

    for i, file in tqdm(enumerate(data_files), desc='Tests', leave=True, total=len(data_files)):
        print(f'\nFile: {data_name}_{i}, {experiment_name}: {experiment_params}')
        df_train, df_test, true_anomalies = get_data_from_json(file.path)
        test_manager.make_tests(window_size, df_train, df_test, true_anomalies)

    end_experiment(metric_repository,
                   plots_repository,
                   model_manager,
                   report_file_name,
                   description + FIT_DETECT)


def single_benchmarking_experiment_single_fit(args: list[str],
                                              experiment_name: str,
                                              model_names: str or list[str],
                                              data_name: str,
                                              directory: os.DirEntry,
                                              hyperparameters: CUSTOM_HYPERPARAMETERS_TYPE,
                                              window_size: int,
                                              experiment_params: str,
                                              save_pdf: bool,
                                              show_plots: bool) -> None:
    (model_manager,
     test_manager,
     metric_repository,
     plots_repository,
     report_file_name,
     description) = init_experiment(args,
                                    experiment_name,
                                    model_names,
                                    objects.SINGLE_ARIMA_DETECT_TIME_MEASURER,
                                    data_name,
                                    hyperparameters,
                                    experiment_params,
                                    save_pdf,
                                    show_plots,
                                    True)
    data_files = get_data_files(directory=directory.path)

    for i, file in tqdm(enumerate(data_files), desc='Tests', leave=True, total=len(data_files)):
        print(f'\nFile: {data_name}_{i}, {experiment_name}: {experiment_params}')
        df_train, df_test, true_anomalies = get_data_from_json(file.path)
        model_manager.fit(df_train)
        test_manager.make_tests(window_size, df_train, df_test, true_anomalies)

    end_experiment(metric_repository,
                   plots_repository,
                   model_manager,
                   report_file_name,
                   description + SINGLE_FIT)
