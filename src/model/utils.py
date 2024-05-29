import pandas as pd

from src.common import constants
from src.common.utils import concat_orion_hyperparameters_dict
from src.constants.custom_types import ORION_HYPERPARAMETERS_TYPE, CUSTOM_HYPERPARAMETERS_TYPE
from src.model.constants import ARIMA
from src.model.proxy import ArimaProxy, OrionModelProxy
from src.orion_applications.hyperparameters.default import default_hyperparameters


def build_proxy(i: int, total_count: int, name: str, hyperparameters: ORION_HYPERPARAMETERS_TYPE) -> OrionModelProxy:
    if name == ARIMA:
        proxy = ArimaProxy
    else:
        proxy = OrionModelProxy
    return proxy(i, total_count, name, hyperparameters)


def get_hyperparameters(hyperparameters: CUSTOM_HYPERPARAMETERS_TYPE, name: str) -> ORION_HYPERPARAMETERS_TYPE:
    default = default_hyperparameters[name]

    if name in hyperparameters.keys():
        main = hyperparameters[name]
    else:
        return default

    return concat_orion_hyperparameters_dict(default, main)


def put_binary_labels(df: pd.DataFrame, name: str, anomalies: pd.DataFrame) -> None:
    df.loc[:, name] = 0

    anomaly = list(anomalies[['start', 'end']].itertuples(index=False))

    for anom in anomaly:
        for i in range(anom[0], anom[1] + constants.DATA_FREQUENCY, constants.DATA_FREQUENCY):
            df.loc[df['timestamp'] == i, name] = 1


def get_y_true_and_y_pred(df_test: pd.DataFrame,
                          true_anomalies: pd.DataFrame,
                          pred_anomalies: pd.DataFrame) -> (pd.Series, pd.Series):
    df_copy = df_test.copy()
    put_binary_labels(df_copy, 'y_true', true_anomalies)
    put_binary_labels(df_copy, 'y_pred', pred_anomalies)

    return df_copy['y_true'], df_copy['y_pred']
