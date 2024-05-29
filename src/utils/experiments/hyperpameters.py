from typing import Any

from src.common.utils import concat_orion_hyperparameters_dict
from src.constants import experiments
from src.constants.custom_types import CUSTOM_HYPERPARAMETERS_TYPE
from src.model import constants


def get_window_size_hyperparameters(**kwargs: Any) -> CUSTOM_HYPERPARAMETERS_TYPE:
    hyperparameters = {
        constants.ARIMA: {
            'mlstars.custom.timeseries_preprocessing.rolling_window_sequences#1': {
                'window_size': kwargs['window_size']},
            'numpy.reshape#1': {
                'newshape': [-1, kwargs['window_size']]},
            'custom.find_anomalies#1': {
                'clustering': kwargs['clustering']}
        },
        constants.LINEAR_AE: {
            'mlstars.custom.timeseries_preprocessing.rolling_window_sequences#1': {
                'window_size': kwargs['window_size']
            },
            'keras.Sequential.DenseSeq2Seq#1': {
                'window_size': kwargs['window_size'],
                'input_shape': [kwargs['window_size'], 1],
                'target_shape': [kwargs['window_size'], 1]
            },
        },
        constants.VAE: {
            'mlstars.custom.timeseries_preprocessing.rolling_window_sequences#1': {
                'window_size': kwargs['window_size'],
            },
        },
        constants.LSTM_AE: {
            'mlstars.custom.timeseries_preprocessing.rolling_window_sequences#1': {
                'window_size': kwargs['window_size']
            },
            'keras.Sequential.LSTMSeq2Seq#1': {
                'window_size': kwargs['window_size'],
                'input_shape': [kwargs['window_size'], 1],
                'target_shape': [kwargs['window_size'], 1]
            },
        },
        constants.LSTM_THRESHOLD: {
            'mlstars.custom.timeseries_preprocessing.rolling_window_sequences#1': {
                'window_size': kwargs['window_size']
            },
        },
        constants.TADGAN: {
            'mlstars.custom.timeseries_preprocessing.rolling_window_sequences#1': {
                'window_size': kwargs['window_size'],
            },
        },
        constants.AER: {
            'mlstars.custom.timeseries_preprocessing.rolling_window_sequences#1': {
                'window_size': kwargs['window_size'],
            },
        }
    }

    return hyperparameters


def get_steps_hyperparameters(**kwargs: Any) -> CUSTOM_HYPERPARAMETERS_TYPE:
    hyperparameters = {
        constants.ARIMA: {
            'mlstars.custom.timeseries_preprocessing.rolling_window_sequences#1': {
                'step_size': kwargs['steps'],
                'target_size': kwargs['steps']},
            'custom.arima#1': {
                'steps': kwargs['steps']},
            'numpy.reshape#2': {
                'newshape': [-1, kwargs['steps']]}
        }
    }

    return hyperparameters


def get_anomaly_size_hyperparameters(**kwargs: Any) -> CUSTOM_HYPERPARAMETERS_TYPE:
    hyperparameters = {
        constants.ARIMA: {
            'custom.find_anomalies#1': {
                'fixed_threshold': False,
                'window_size': kwargs['window_size_find_anomalies']}

        }
    }

    return hyperparameters


def concat_custom_hyperparameters(minor_priority: CUSTOM_HYPERPARAMETERS_TYPE,
                                  major_priority: CUSTOM_HYPERPARAMETERS_TYPE) -> CUSTOM_HYPERPARAMETERS_TYPE:
    hyperparameters = dict()
    for model_name in major_priority.keys():
        hyperparameters[model_name] = concat_orion_hyperparameters_dict(minor_priority[model_name],
                                                                        major_priority[model_name])

    return hyperparameters


def get_experiment_hyperparameters(experiment_name: str, **kwargs: Any) -> CUSTOM_HYPERPARAMETERS_TYPE:
    if 'window_size' not in kwargs.keys() or 'clustering' not in kwargs.keys() or (
            experiment_name == experiments.STEPS and 'steps' not in kwargs.keys()) or (
            experiment_name == experiments.ANOMALY_SIZE and 'window_size_find_anomalies' not in kwargs.keys()):
        raise ValueError('Not enough parameters for getting valid hyperparameters')

    window_size_hyperparameters = get_window_size_hyperparameters(**kwargs)

    if experiment_name == experiments.WINDOW_SIZE:
        return window_size_hyperparameters
    elif experiment_name == experiments.STEPS:
        hyperparameters = get_steps_hyperparameters(**kwargs)
    elif experiment_name == experiments.ANOMALY_SIZE:
        hyperparameters = get_anomaly_size_hyperparameters(**kwargs)
    else:
        raise ValueError(f'Unknown experiment: {experiment_name}')

    return concat_custom_hyperparameters(window_size_hyperparameters, hyperparameters)
