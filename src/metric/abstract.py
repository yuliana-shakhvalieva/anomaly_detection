from dataclasses import dataclass
from typing import NoReturn

import pandas as pd

from src.constants.custom_types import CLASSIFICATION_REPORT_TYPE


@dataclass
class Metric:
    id: str
    value: float or CLASSIFICATION_REPORT_TYPE


@dataclass
class AnomaliesCount:
    true: int
    pred: int


class MetricBase:
    def __init__(self, id: str) -> None:
        self.id = id

    def _create_metric(self, value: float or dict[str, dict]) -> list[Metric]:
        return [Metric(self.id, value)]


class MetricCalculator(MetricBase):
    def calculate(self,
                  y_true: pd.Series,
                  y_pred: pd.Series,
                  anomalies_count: AnomaliesCount) -> NoReturn:
        raise NotImplementedError

    def _calculate_ratio(self, value: int, anomalies_count: AnomaliesCount) -> float:
        return value / self._get_divider(anomalies_count)

    def _get_divider(self, anomalies_count: AnomaliesCount) -> NoReturn:
        raise NotImplementedError
