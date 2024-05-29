from typing import Callable

import pandas as pd

from src.common.utils import measure_seconds
from src.metric.abstract import MetricBase, Metric
from src.metric.constants import names
from src.model.constants import ARIMA
from src.model.proxy import OrionModelProxy
from src.report.creation.utils import get_window_size_rolling_window_sequences


class ModelTimeMeasurer(MetricBase):
    def __init__(self, metric_name) -> None:
        super().__init__(metric_name)

    def measure_time(self,
                     model: OrionModelProxy,
                     df_test: pd.DataFrame,
                     df_train: pd.DataFrame or None) -> (pd.DataFrame, list[Metric]):
        raise NotImplementedError

    def _measure_time(self, func: Callable, df: pd.DataFrame) -> (pd.DataFrame, list[Metric]):
        pred_anomalies, spent_time = measure_seconds(func, df)
        return pred_anomalies, super()._create_metric(spent_time)


class ModelDetectTimeMeasurer(ModelTimeMeasurer):
    def __init__(self) -> None:
        super().__init__(names.DETECT_TIME)

    def measure_time(self,
                     model: OrionModelProxy,
                     df_test: pd.DataFrame,
                     df_train: pd.DataFrame or None) -> (pd.DataFrame, list[Metric]):
        if df_train is not None:
            model.fit(df_train)

        func = getattr(model, 'detect')
        return super()._measure_time(func, df_test)


class ModelFitDetectTimeMeasurer(ModelTimeMeasurer):
    def __init__(self) -> None:
        super().__init__(names.FIT_DETECT_TIME)

    def measure_time(self,
                     model: OrionModelProxy,
                     df_test: pd.DataFrame,
                     df_train: pd.DataFrame or None) -> (pd.DataFrame, list[Metric]):
        func = getattr(model, 'fit_detect')
        return super()._measure_time(func, df_test)


class ArimaModelCountMeasurer(MetricBase):
    def __init__(self, time_measurer: ModelTimeMeasurer) -> None:
        super().__init__(names.ARIMA_MODELS_COUNT)
        self.time_measurer = time_measurer

    def measure_time(self,
                     model: OrionModelProxy,
                     df_test: pd.DataFrame,
                     df_train: pd.DataFrame or None) -> (pd.DataFrame, list[Metric]):
        if model.name != ARIMA:
            return self.time_measurer.measure_time(model, df_test, df_train)

        pred_anomalies, metrics = self.time_measurer.measure_time(model, df_test, df_train)
        window_size = get_window_size_rolling_window_sequences(model.model)
        steps = model.model_hyperparameters['steps']
        count_models = (df_test.shape[0] - window_size) // steps
        metrics.extend(self._create_metric(count_models))
        return pred_anomalies, metrics


class SingleArimaDetectTimeMeasurer(MetricBase):
    def __init__(self, arima_model_count_measurer: ArimaModelCountMeasurer) -> None:
        super().__init__(names.ARIMA_DETECT_TIME)
        self.arima_model_count_measurer = arima_model_count_measurer

    def measure_time(self,
                     model: OrionModelProxy,
                     df_test: pd.DataFrame,
                     df_train: pd.DataFrame or None) -> (pd.DataFrame, list[Metric]):
        if model.name != ARIMA:
            return self.arima_model_count_measurer.measure_time(model, df_test, df_train)

        pred_anomalies, metrics = self.arima_model_count_measurer.measure_time(model, df_test, df_train)
        detect_time, count_models = metrics[0].value, metrics[1].value
        metrics.extend(self._create_metric(detect_time / count_models))
        return pred_anomalies, metrics
