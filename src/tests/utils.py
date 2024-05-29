from typing import Any

import pandas as pd

from src.metric.abstract import Metric
from src.model.manager import ModelManager
from src.tests.plots_requirement import PlotsRequirements
from src.utils.generation import put_anomalies


def single_test_on_full_df_test(_,
                                model_manager: ModelManager,
                                df_test: pd.DataFrame,
                                anomaly_generation_params: dict[str, Any] or None,
                                true_anomalies: pd.DataFrame or None = None) -> ([str, list[Metric]], PlotsRequirements):
    if true_anomalies is None:
        df_test, true_anomalies = put_anomalies(df_test, anomaly_generation_params)

    model_metrics, model_pred_anomalies = model_manager.test_on_full_df_test(df_test, true_anomalies)

    return model_metrics, PlotsRequirements(df_test=df_test,
                                            true_anomalies=true_anomalies,
                                            model_pred_anomalies=model_pred_anomalies)


def single_test(model_manager: ModelManager,
                window_size: int,
                df_train: pd.DataFrame,
                df_test: pd.DataFrame,
                true_anomalies: pd.DataFrame,
                make_fit: bool) -> ([str, list[Metric]], PlotsRequirements):
    model_metrics, model_pred_anomalies = model_manager.test(window_size, df_train, df_test, true_anomalies, make_fit)

    return model_metrics, PlotsRequirements(df_test=df_test,
                                            true_anomalies=true_anomalies,
                                            model_pred_anomalies=model_pred_anomalies)
