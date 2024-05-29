import numpy as np
from orion.primitives import timeseries_anomalies
from sklearn.cluster import DBSCAN


def clustering_find_anomalies(errors: np.ndarray,
                              index: np.ndarray,
                              anomaly_padding: int) -> np.ndarray:
    X = errors.reshape(len(errors), -1)
    clustering = DBSCAN(eps=0.1, min_samples=3)
    labels = clustering.fit_predict(X)

    binary_anomalies = [0 for _ in range(len(labels))]

    for position, label in enumerate(labels):
        if label == -1:
            for j in range(max(0, position - anomaly_padding), min(position + anomaly_padding + 1, len(labels))):
                binary_anomalies[j] = 1

    anomalies = []
    start = None
    i = 0
    while i < len(binary_anomalies):
        if binary_anomalies[i] == 1 and start is None:
            start = i
            i += anomaly_padding * 2
        elif binary_anomalies[i] == 0 and binary_anomalies[i - 1] == 1 and start is not None:
            end = i
            anomalies.append([index[int(start)], index[int(end)], -1])
            start = None
            i += 1
        else:
            i += 1

    return np.asarray(anomalies)


def orion_find_anomalies(errors: np.ndarray,
                         index: np.ndarray,
                         z_range: list[int],
                         window_size: int,
                         window_size_portion: float,
                         window_step_size: int,
                         window_step_size_portion: float,
                         min_percent: float,
                         anomaly_padding: int,
                         lower_threshold: bool,
                         fixed_threshold: bool,
                         inverse: bool) -> np.ndarray:
    return timeseries_anomalies.find_anomalies(errors, index, z_range, window_size, window_size_portion,
                                               window_step_size, window_step_size_portion, min_percent,
                                               anomaly_padding, lower_threshold, fixed_threshold, inverse)


def find_anomalies(errors: np.ndarray,
                   index: np.ndarray,
                   clustering: bool = False,
                   z_range: list[int] = (0, 10),
                   window_size: int = None,
                   window_size_portion: float = None,
                   window_step_size: int = None,
                   window_step_size_portion: float = None,
                   min_percent: float = 0.1,
                   anomaly_padding: int = 50,
                   lower_threshold: bool = False,
                   fixed_threshold: bool = None,
                   inverse: bool = False) -> np.ndarray:
    if clustering:
        return clustering_find_anomalies(errors, index, anomaly_padding)

    return orion_find_anomalies(errors, index, z_range, window_size, window_size_portion,
                                window_step_size, window_step_size_portion, min_percent,
                                anomaly_padding, lower_threshold, fixed_threshold, inverse)
