from functools import partial
from multiprocessing import Pool
from typing import Any

import pandas as pd
from tqdm import trange, tqdm

from src.common import constants
from src.constants.counting_way import MULTIPROCESS_MODE, SEQUENTIAL_MODE, CPU_COUNT
from src.metric.abstract import Metric
from src.metric.repository import MetricRepository
from src.model.manager import ModelManager
from src.tests.plots_requirement import PlotsRequirements
from src.tests.utils import single_test_on_full_df_test, single_test
from src.utils.decorators.time import measure_time
from src.utils.experiments.command_line import get_command_line_params
from src.visualization.repository import PlotsRepository


class TestsManager:
    def __init__(self,
                 args: list[str],
                 model_manager: ModelManager,
                 metric_repository: MetricRepository,
                 plots_repository: PlotsRepository) -> None:
        self.counting_way, self.chunk_size = get_command_line_params(args)
        self.model_manager = model_manager
        self.metric_repository = metric_repository
        self.plots_repository = plots_repository

    def make_tests(self,
                   window_size: int,
                   df_train: pd.DataFrame,
                   df_test: pd.DataFrame,
                   true_anomalies: pd.DataFrame,
                   make_fit: bool = False) -> None:
        model_metrics, plots_requirements = single_test(self.model_manager,
                                                        window_size,
                                                        df_train,
                                                        df_test,
                                                        true_anomalies,
                                                        make_fit)
        self.__save_results(model_metrics, plots_requirements)

    @measure_time('All tests')
    def make_tests_on_full_df_test(self,
                                   df_test: pd.DataFrame,
                                   anomaly_generation_params: dict[str, Any] or None = None,
                                   true_anomalies: pd.DataFrame or None = None) -> None:

        if true_anomalies is not None:
            model_metrics, plots_requirements = single_test_on_full_df_test(self,
                                                                            self.model_manager,
                                                                            df_test,
                                                                            anomaly_generation_params,
                                                                            true_anomalies)
            self.__save_results(model_metrics, plots_requirements)

        elif self.counting_way == MULTIPROCESS_MODE:
            self.__make_multiprocessing_tests_on_full_df_test(df_test, anomaly_generation_params)
        elif self.counting_way == SEQUENTIAL_MODE:
            self.__make_sequential_tests_on_full_df_test(df_test, anomaly_generation_params)

    def __make_multiprocessing_tests_on_full_df_test(self,
                                                     df_test: pd.DataFrame,
                                                     anomaly_generation_params: dict[str, Any]) -> None:
        with Pool(CPU_COUNT) as pool:
            list_results = tqdm(pool.imap_unordered(
                partial(single_test_on_full_df_test,
                        model_manager=self.model_manager,
                        df_test=df_test,
                        anomaly_generation_params=anomaly_generation_params),
                range(constants.NUM_TEST)
            ), desc='Multiprocessing tests', leave=False, total=constants.NUM_TEST)

            for model_metrics, plots_requirements in list_results:
                self.__save_results(model_metrics, plots_requirements)

    def __make_sequential_tests_on_full_df_test(self,
                                                df_test: pd.DataFrame,
                                                anomaly_generation_params: dict[str, Any]) -> None:
        for _ in trange(constants.NUM_TEST, desc='Sequential tests', leave=False):
            model_metrics, plots_requirements = single_test_on_full_df_test(self,
                                                                            self.model_manager,
                                                                            df_test,
                                                                            anomaly_generation_params)
            self.__save_results(model_metrics, plots_requirements)

    def __save_results(self, model_metrics: [str, list[Metric]], plots_requirements: PlotsRequirements) -> None:
        self.metric_repository.save(model_metrics)
        self.plots_repository.save(plots_requirements)
