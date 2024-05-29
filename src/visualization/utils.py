from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure


def convert_date(time_list: pd.Series) -> list[datetime]:
    converted = list()
    for x in time_list:
        converted.append(datetime.fromtimestamp(x))
    return converted


def convert_date_single(x: int) -> datetime:
    return datetime.fromtimestamp(x)


def get_figure(df: pd.DataFrame, *,
               true_anomalies: pd.DataFrame = pd.DataFrame(columns=['start', 'end']),
               pred_anomalies: pd.DataFrame = pd.DataFrame(columns=['start', 'end']),
               title: str or None = None,
               orion_format: bool = True,
               show_figure: bool = False) -> Figure:
    if orion_format:
        time = convert_date(df['timestamp'])
    else:
        time = df['timestamp']

    figure = plt.figure(figsize=(20, 5))
    plt.plot(time, df['value'])

    for anomaly, color, label in zip([true_anomalies, pred_anomalies],
                                     ['green', 'red'],
                                     ['true anomalies', 'pred anomalies']):

        if not isinstance(anomaly, list):
            anomaly = list(anomaly[['start', 'end']].itertuples(index=False))

        for i, anom in enumerate(anomaly):
            t1, t2 = anom

            if orion_format:
                t1 = convert_date_single(t1)
                t2 = convert_date_single(t2)

            plt.axvspan(t1, t2, color=color, alpha=0.2, label='_' * i + label)

    if title is not None:
        plt.title(title, size=16)

    plt.legend()

    if show_figure:
        plt.show()

    plt.close()

    return figure
