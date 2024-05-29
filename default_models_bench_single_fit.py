import sys
import warnings

from src.constants import experiments
from src.model import constants
from src.utils.decorators.time import measure_time
from src.utils.experiments.hyperpameters import get_experiment_hyperparameters
from src.utils.experiments import pipeline
from src.utils.experiments.single import single_benchmarking_experiment_single_fit

warnings.filterwarnings('ignore')


@measure_time('All experiments')
def main(args: list[str]) -> None:
    save_pdf = True
    show_plots = False
    find_anomalies_with_clustering = False

    experiment_name = experiments.WINDOW_SIZE
    model_names = constants.DEFAULT_MODELS

    for window_size_rolling_window_sequences in [100]:
        hyperparameters = get_experiment_hyperparameters(experiment_name,
                                                         window_size=window_size_rolling_window_sequences,
                                                         clustering=find_anomalies_with_clustering)

        experiment_params = f'{window_size_rolling_window_sequences}'

        for directory in pipeline.get_data_files(bench=True, directory='benchmark/data', custom_file_names='messages'):
            data_name = directory.name

            single_benchmarking_experiment_single_fit(args,
                                                      experiment_name,
                                                      model_names,
                                                      data_name,
                                                      directory,
                                                      hyperparameters,
                                                      window_size_rolling_window_sequences,
                                                      experiment_params,
                                                      save_pdf,
                                                      show_plots)


if __name__ == '__main__':
    main(sys.argv)
