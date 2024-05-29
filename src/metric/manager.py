import pandas as pd

from src.metric.abstract import AnomaliesCount, MetricCalculator, Metric
from src.metric.measurers import SingleArimaDetectTimeMeasurer
from src.metric.utils.utils import flatten
from src.model.proxy import OrionModelProxy


class MetricManager:
    def __init__(self,
                 metric_calculators: list[MetricCalculator],
                 metric_invoke_measurer: SingleArimaDetectTimeMeasurer) -> None:
        self.metric_calculators = metric_calculators
        self.metric_invoke_measurer = metric_invoke_measurer

    def calculate(self,
                  y_true: pd.Series,
                  y_pred: pd.Series,
                  anomalies_count: AnomaliesCount) -> list[Metric]:
        metrics = [calculator.calculate(y_true, y_pred, anomalies_count) for calculator in self.metric_calculators]
        return flatten(metrics)

    def measure_over_invoke(self,
                            model: OrionModelProxy,
                            df_test: pd.DataFrame,
                            df_train: pd.DataFrame or None = None) -> (pd.DataFrame, list[Metric]):
        return self.metric_invoke_measurer.measure_time(model, df_test, df_train)
