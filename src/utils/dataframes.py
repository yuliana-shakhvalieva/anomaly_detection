import random

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

from src.common.constants import NUM_SEQUENCES, LEN_SEQUENCE
from src.common.utils import prepare_path
from src.utils.decorators.loggers import log
from src.utils.print import pretty_print, get_pretty_title
from src.visualization.utils import get_figure


def read_df(file_name: str) -> pd.DataFrame:
    df = pd.read_csv(file_name, sep=',')
    df = df.drop(columns=['Label'])
    df.rename(columns={'Date': 'timestamp', 'Value': 'value'}, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def get_diff(df: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    df_diff = df
    df_diff.value = df.value.diff(periods=periods)
    df_diff = df_diff.dropna()
    return df_diff


def analyze_df(df: pd.DataFrame) -> None:
    shape = df.shape
    pretty_print('Размер датасета:', shape)

    dtypes = df.dtypes
    pretty_print('Типы данных:', dtypes)

    statistics = df.describe()
    pretty_print('Описательные статистики:', statistics)

    print(f'\n\n{get_pretty_title("График временного ряда:")}')

    plt.figure(figsize=(20, 5))
    plt.plot(df['timestamp'], df['value'])
    plt.show()

    print(f'\n\n{get_pretty_title("Коррелограмма:")}')
    plt.figure(figsize=(20, 5))
    plot_acf(df['value'], lags=35, ax=plt.gca())
    plt.title('ACF')
    plt.show()

    plt.figure(figsize=(20, 5))
    plot_pacf(df['value'], lags=35, ax=plt.gca())
    plt.title('PACF')
    plt.show()

    adf = adfuller(df['value'])

    adf_statistic = adf[0]
    p_value = adf[1]

    stationary = 'Ряд не стационарен на 5% уровне значимости'
    alpha = 0.05

    if p_value < alpha:
        stationary = 'Ряд стационарен на 5% уровне значимости.'

    adf_results = f'ADF statistic: {adf_statistic}\nP-value: {p_value}\n\n{stationary}'
    pretty_print('Тест Дики-Фуллера:', adf_results)


def orion_format(df: pd.DataFrame) -> pd.DataFrame:
    dt_time = df.timestamp
    df.timestamp = (dt_time - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')
    return df


def apply_orion_format(df_train: pd.DataFrame, df_test: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    df_train = df_train.apply(orion_format, axis=1)
    df_test = df_test.apply(orion_format, axis=1)
    return df_train, df_test


def split_data_by_date(df: pd.DataFrame,
                       data: str = '2023-12-24',
                       orion_format: bool = False,
                       show_plots: bool = True) -> (pd.DataFrame, pd.DataFrame):
    if orion_format:
        orion_data = (pd.Timestamp(data) - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')
        df_train = df[df['timestamp'] < orion_data]
        df_test = df[df['timestamp'] >= orion_data]
    else:
        df_train = df[df['timestamp'].dt.date < pd.Timestamp(data).date()]
        df_test = df[df['timestamp'].dt.date >= pd.Timestamp(data).date()]

    if show_plots:
        get_figure(df_train, title='Train data', orion_format=orion_format).show()
        get_figure(df_test, title='Test data', orion_format=orion_format).show()

    return df_train, df_test


@log(text='Splitting data...')
def get_test_train(df: pd.DataFrame, show_plots: bool = False) -> (pd.DataFrame, pd.DataFrame):
    df_train, df_test = split_data_by_date(df, '2024-01-18', show_plots=show_plots)
    df_train, df_test = apply_orion_format(df_train, df_test)

    return df_train, df_test


def save_data_split_by_unique_label(input_file_name: str, output_path: str) -> None:
    prepare_path(output_path)

    df = pd.read_csv(input_file_name, sep=';')
    unique_labels = df['Label'].unique()

    for label in unique_labels:
        df[df['Label'] == label].to_csv(output_path + f'{label}.csv', sep=',', index=False, encoding='utf-8')


@log(text='Getting random sequences...')
def get_random_sequences_from_data(df: pd.DataFrame) -> list[pd.DataFrame]:
    sequences = []

    for _ in range(NUM_SEQUENCES):
        start_i = random.randint(0, df.shape[0] - 1 - LEN_SEQUENCE)
        sequences.append(df.iloc[start_i: start_i + LEN_SEQUENCE])

    return sequences
