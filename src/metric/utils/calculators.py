import numpy as np
import pandas as pd

from src.metric.utils.utils import anomaly_detected, get_perfect_range


def calculate_iou(y_true: pd.Series, y_pred: pd.Series) -> float:
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))

    iou = true_positive / (true_positive + false_negative + false_positive)

    if iou == np.nan:
        iou = 1.0

    return iou


def calculate_good_detection(y_true: pd.Series, y_pred: pd.Series) -> int:
    good_detection, i = 0, 0

    while i < len(y_true):
        start, end = get_perfect_range(i, len(y_true))

        if (anomaly_detected(y_true, i) and
                not anomaly_detected(y_true, i - 1) and
                anomaly_detected(y_pred, [start, end])):
            good_detection += 1

        i += 1

    return good_detection


def calculate_errors(y_detected: pd.Series, y_not_detected: pd.Series) -> int:
    metric, i = 0, 0
    start = None

    while i < len(y_detected):
        if (anomaly_detected(y_detected, i) and
                start is None):

            start = i

        elif (anomaly_detected(y_detected, i - 1) and
              not anomaly_detected(y_detected, i) and
              start is not None):

            end = i
            if not anomaly_detected(y_not_detected, [start, end]):
                metric += 1
            start = None

        i += 1

    return metric


def calculate_late_detection(y_true: pd.Series, y_pred: pd.Series) -> int:
    late_detection, i = 0, 0
    potential_late_detection = False

    while i < len(y_pred):
        start, end = get_perfect_range(i, len(y_true))

        if (not anomaly_detected(y_true, i - 1) and
                anomaly_detected(y_true, i) and
                not anomaly_detected(y_pred, [start, end]) and
                not potential_late_detection):

            potential_late_detection = True
            i = end

        elif (anomaly_detected(y_true, i) and
              potential_late_detection and
              anomaly_detected(y_pred, i)):

            late_detection += 1
            potential_late_detection = False
            i += 1

        elif (not anomaly_detected(y_true, i) and
              potential_late_detection):

            potential_late_detection = False
            i += 1

        else:
            i += 1

    return late_detection
