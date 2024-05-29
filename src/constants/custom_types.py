from typing import Any

from matplotlib.figure import Figure

ORION_HYPERPARAMETERS_TYPE = dict[str, dict[str, Any]]
CUSTOM_HYPERPARAMETERS_TYPE = dict[str, ORION_HYPERPARAMETERS_TYPE]
CLASSIFICATION_REPORT_TYPE = dict[str, dict[str, float] or list[float] or float]
METRIC_REPOSITORY_TYPE = dict[str, dict[str, list[CLASSIFICATION_REPORT_TYPE or float or int]
                                             or CLASSIFICATION_REPORT_TYPE or float]]
PLOT_REPOSITORY_TYPE = dict[str, list[Figure]]
