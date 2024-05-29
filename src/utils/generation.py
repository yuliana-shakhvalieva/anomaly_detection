import random
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as sts

from src.common import constants
from src.constants import generation
from src.utils.decorators.loggers import log


@log('Generating data...')
def generate_data() -> pd.DataFrame:
    data_length = generation.COUNT_DAYS * 24 * 60 * 60 // constants.DATA_FREQUENCY
    x = np.linspace(0, 2 * np.pi, data_length)
    data = np.sin(generation.COUNT_DAYS * x) * generation.SCALE + generation.ADD

    norm_rv = sts.norm(loc=generation.MEAN, scale=generation.STD)
    noise = norm_rv.rvs(size=data_length)
    data += noise

    start_data = pd.Timestamp('2024-01-15')
    end_data = start_data + pd.DateOffset(generation.COUNT_DAYS)
    date_range = pd.date_range(start=start_data, end=end_data, freq=constants.FREQ)

    df = pd.DataFrame({'timestamp': date_range})[:data_length]
    df['value'] = data

    return df


def try_put_anomalies(df: pd.DataFrame,
                      p_anomaly: float,
                      possible_anomaly_length: list[int],
                      start_from: int) -> (pd.DataFrame, pd.DataFrame):
    df_copy = df.copy()
    true_anomalies = pd.DataFrame(columns=['start', 'end'])
    data_length = df_copy.shape[0]

    columns = df_copy.columns.tolist()
    timestamp_idx = columns.index('timestamp')
    value_idx = columns.index('value')

    mean = df_copy['value'].mean() // 20
    std = df_copy['value'].std() // 20

    if std < 500:
        curvature_coefficient = 0.13
    else:
        curvature_coefficient = 0.23

    if random.random() < 0.5:
        direction = 1  # аномалия вверх
    else:
        direction = -1  # аномалия вниз

    i = 0
    while i != data_length:
        if data_length - i < min(possible_anomaly_length):
            break
        elif i >= start_from and random.random() < p_anomaly:
            anomaly_length = random.choice(possible_anomaly_length)

            if i + anomaly_length >= data_length:
                anomaly_length = data_length - i - 1

            if anomaly_length % 2 != 0:
                anomaly_length -= 1

            true_anomalies.loc[true_anomalies.shape[0]] = [df_copy.iloc[i, timestamp_idx],
                                                           df_copy.iloc[i + anomaly_length, timestamp_idx]]

            norm_rv = sts.norm(loc=mean, scale=std)
            noises = norm_rv.rvs(size=anomaly_length)

            anomaly_coefficients = np.linspace(0, curvature_coefficient, anomaly_length // 2)

            for idx, noise in enumerate(noises):
                if idx < anomaly_length // 2:
                    anomaly_coefficient = 1 + direction * anomaly_coefficients[idx]
                else:
                    anomaly_coefficient = 1 + direction * anomaly_coefficients[(anomaly_length - idx - 1)]

                df_copy.iloc[i + idx, value_idx] *= anomaly_coefficient
                df_copy.iloc[i + idx, value_idx] += direction * noise

            i += anomaly_length

        else:
            i += 1

    return df_copy, true_anomalies


def put_anomalies(df: pd.DataFrame, anomaly_generation_params: dict[str, Any]) -> (pd.DataFrame, pd.DataFrame):
    have_anomalies = False
    df_with_anomalies, true_anomalies = None, None

    while not have_anomalies:
        df_with_anomalies, true_anomalies = try_put_anomalies(df, **anomaly_generation_params)
        if true_anomalies.shape[0] != 0:
            have_anomalies = True

    return df_with_anomalies, true_anomalies
