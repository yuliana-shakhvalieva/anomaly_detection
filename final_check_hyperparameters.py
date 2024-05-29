import sys
import warnings
from typing import Any

from src.constants import experiments
from src.model import constants
from src.utils.decorators.time import measure_time
from src.utils.experiments.hyperpameters import get_experiment_hyperparameters
from src.utils.experiments import pipeline
from src.utils.experiments.single import single_benchmarking_experiment_fit_on_history_data

warnings.filterwarnings('ignore')

BASIC_HYPERPARAMETERS = {
    'account': {
        'window_size': 50,
        'steps': 3,
        'window_size_find_anomalies': 2000,
    },
    'messages': {
        'window_size': 50,
        'steps': 2,
        'window_size_find_anomalies': 2000,
    },
    'statEvents': {
        'window_size': 70,
        'steps': 1,
        'window_size_find_anomalies': 2000,
    },
    'stats': {
        'window_size': 90,
        'steps': 1,
        'window_size_find_anomalies': 2000,
    },
    'users': {
        'window_size': 90,
        'steps': 2,
        'window_size_find_anomalies': 2000,
    },
}


def get_arima_hyperparameters(sequence_name: str, find_anomalies_with_clustering: bool) -> dict[str, dict[str, Any]]:
    hyperparameters = BASIC_HYPERPARAMETERS[sequence_name]

    return {constants.ARIMA: {
        'mlstars.custom.timeseries_preprocessing.rolling_window_sequences#1': {
            'window_size': hyperparameters['window_size'],
            'step_size': hyperparameters['steps'],
            'target_size': hyperparameters['steps'],
        },
        'numpy.reshape#1': {
            'newshape': [-1, hyperparameters['window_size']],
        },
        'custom.find_anomalies#1': {
            'fixed_threshold': False,
            'window_size': hyperparameters['window_size_find_anomalies'],
            'clustering': find_anomalies_with_clustering,
        },
        'custom.arima#1': {
            'steps': hyperparameters['steps'],
        },
        'numpy.reshape#2': {
            'newshape': [-1, hyperparameters['steps']],
        },
    }
    }


@measure_time('All experiments')
def main(args: list[str]) -> None:
    save_pdf = True
    show_plots = False
    find_anomalies_with_clustering = False

    experiment_name = experiments.WINDOW_SIZE
    model_names = constants.ARIMA

    for directory in pipeline.get_data_files(bench=True, directory='benchmark/data'):
        data_name = directory.name
        hyperparameters = get_arima_hyperparameters(data_name, find_anomalies_with_clustering)

        single_benchmarking_experiment_fit_on_history_data(args,
                                                           experiment_name,
                                                           model_names,
                                                           data_name,
                                                           directory,
                                                           hyperparameters,
                                                           BASIC_HYPERPARAMETERS[data_name]['window_size'],
                                                           'experiment_params',
                                                           save_pdf,
                                                           show_plots)


if __name__ == '__main__':
    main(sys.argv)
