import pandas as pd
from sklearn.metrics import classification_report

from src.metric.abstract import MetricCalculator, Metric, AnomaliesCount
from src.metric.constants import names
from src.metric.utils import calculators


class IoUCalculator(MetricCalculator):
    def __init__(self) -> None:
        super().__init__(names.IOU)

    def calculate(self,
                  y_true: pd.Series,
                  y_pred: pd.Series,
                  anomalies_count: AnomaliesCount) -> list[Metric]:
        iou = calculators.calculate_iou(y_true, y_pred)
        return self._create_metric(iou)


class GoodDetectionCalculator(MetricCalculator):
    def __init__(self) -> None:
        super().__init__(names.GOOD_DETECTION)

    def calculate(self,
                  y_true: pd.Series,
                  y_pred: pd.Series,
                  anomalies_count: AnomaliesCount) -> list[Metric]:
        good_detection = calculators.calculate_good_detection(y_true, y_pred)
        ratio = self._calculate_ratio(good_detection, anomalies_count)
        return self._create_metric(ratio)

    def _get_divider(self, anomalies_count: AnomaliesCount) -> int:
        return anomalies_count.true


class FalsePositiveCalculator(MetricCalculator):
    def __init__(self) -> None:
        super().__init__(names.FALSE_POSITIVES)

    def calculate(self,
                  y_true: pd.Series,
                  y_pred: pd.Series,
                  anomalies_count: AnomaliesCount) -> list[Metric]:
        false_positives = calculators.calculate_errors(y_detected=y_pred, y_not_detected=y_true)
        ratio = self._calculate_ratio(false_positives, anomalies_count)
        return self._create_metric(ratio)

    def _calculate_ratio(self, value: int, anomalies_count: AnomaliesCount) -> float:
        if self._get_divider(anomalies_count) == 0:
            return 0.0
        return super()._calculate_ratio(value, anomalies_count)

    def _get_divider(self, anomalies_count: AnomaliesCount) -> int:
        return anomalies_count.pred


class LateDetectionCalculator(MetricCalculator):
    def __init__(self) -> None:
        super().__init__(names.LATE_DETECTION)

    def calculate(self,
                  y_true: pd.Series,
                  y_pred: pd.Series,
                  anomalies_count: AnomaliesCount) -> list[Metric]:
        late_detection = calculators.calculate_late_detection(y_true, y_pred)
        ratio = self._calculate_ratio(late_detection, anomalies_count)
        return self._create_metric(ratio)

    def _get_divider(self, anomalies_count: AnomaliesCount) -> int:
        return anomalies_count.true


class NotDetectedCalculator(MetricCalculator):
    def __init__(self) -> None:
        super().__init__(names.NOT_DETECTED)

    def calculate(self,
                  y_true: pd.Series,
                  y_pred: pd.Series,
                  anomalies_count: AnomaliesCount) -> list[Metric]:
        not_detected = calculators.calculate_errors(y_detected=y_true, y_not_detected=y_pred)
        ratio = self._calculate_ratio(not_detected, anomalies_count)
        return self._create_metric(ratio)

    def _get_divider(self, anomalies_count: AnomaliesCount) -> int:
        return anomalies_count.true


class ClassificationReportCalculator(MetricCalculator):
    def __init__(self) -> None:
        super().__init__(names.CLASSIFICATION_REPORT)

    def calculate(self,
                  y_true: pd.Series,
                  y_pred: pd.Series,
                  anomalies_count: AnomaliesCount) -> list[Metric]:
        report = classification_report(y_true, y_pred, output_dict=True)
        return self._create_metric(report)
