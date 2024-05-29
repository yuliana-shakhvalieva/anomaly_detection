import sys
import warnings

from src.constants import experiments
from src.model import constants
from src.utils.dataframes import get_random_sequences_from_data, read_df
from src.utils.decorators.time import measure_time
from src.utils.experiments import pipeline
from src.utils.experiments.hyperpameters import get_experiment_hyperparameters
from src.utils.experiments.single import single_random_experiment

warnings.filterwarnings('ignore')


@measure_time('All experiments')
def main(args: list[str]) -> None:
    save_pdf = True
    show_plots = False
    find_anomalies_with_clustering = True

    experiment_name = experiments.STEPS
    model_names = [constants.LINEAR_AE]
    best_window_sizes = pipeline.get_best_window_size_rolling_window_sequences('best_window_sizes_clustering.csv')

    for file in pipeline.get_data_files():
        sequences = get_random_sequences_from_data(read_df(file.path))
        data_name = file.name

        window_sizes = best_window_sizes[data_name]
        for window_size_rolling_window_sequences in window_sizes:
            for steps in [1, 2, 3, 4, 5]:
                hyperparameters = get_experiment_hyperparameters(experiment_name,
                                                                 window_size=window_size_rolling_window_sequences,
                                                                 clustering=find_anomalies_with_clustering,
                                                                 steps=steps)

                experiment_params = f'{window_size_rolling_window_sequences}_{steps}'

                anomaly_generation_params = pipeline.get_anomaly_generation_params(
                    start_from=window_size_rolling_window_sequences)

                single_random_experiment(args,
                                         experiment_name,
                                         model_names,
                                         data_name,
                                         sequences,
                                         hyperparameters,
                                         anomaly_generation_params,
                                         experiment_params,
                                         save_pdf,
                                         show_plots)


if __name__ == '__main__':
    main(sys.argv)
