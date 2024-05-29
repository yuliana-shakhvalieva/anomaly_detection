import sys
import warnings

from src.constants import experiments
from src.utils.decorators.time import measure_time
from src.utils.experiments.hyperpameters import get_experiment_hyperparameters
from src.utils.experiments import pipeline
from src.utils.experiments.single import single_benchmarking_experiment_fit_detect

warnings.filterwarnings('ignore')


@measure_time('All experiments')
def main(args: list[str]) -> None:
    save_pdf = True
    show_plots = False
    find_anomalies_with_clustering = False

    experiment_name = experiments.ANOMALY_SIZE
    best_window_sizes = pipeline.get_best_window_size_rolling_window_sequences('best_window_sizes_fit_detect.csv')

    for directory in pipeline.get_data_files(bench=True, directory='benchmark/data'):
        data_name = directory.name
        window_sizes = best_window_sizes[data_name]

        for window_size_rolling_window_sequences in window_sizes:
            for part in [0.33, 0.5, 1]:
                window_size_find_anomalies = int(12 // 2 * part)

                hyperparameters = get_experiment_hyperparameters(experiment_name,
                                                                 window_size=window_size_rolling_window_sequences,
                                                                 clustering=find_anomalies_with_clustering,
                                                                 window_size_find_anomalies=window_size_find_anomalies)

                experiment_params = f'{window_size_rolling_window_sequences}_{part}'

                single_benchmarking_experiment_fit_detect(args,
                                                          experiment_name,
                                                          data_name,
                                                          directory,
                                                          hyperparameters,
                                                          window_size_rolling_window_sequences,
                                                          experiment_params,
                                                          save_pdf,
                                                          show_plots)


if __name__ == '__main__':
    main(sys.argv)