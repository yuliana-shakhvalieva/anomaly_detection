import json

import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from tqdm.auto import tqdm

from src.common.constants import LEN_SEQUENCE, NUM_TEST
from src.common.utils import prepare_path
from src.utils.dataframes import read_df, get_random_sequences_from_data, orion_format
from src.utils.experiments.pipeline import get_anomaly_generation_params, get_data_files
from src.utils.generation import put_anomalies
from src.visualization.utils import get_figure


def get_path(*args: str) -> str:
    path = '/'.join(args)
    prepare_path(path)

    return path


def get_data(sequence: pd.DataFrame, params: dict) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    df_train = sequence.iloc[:(LEN_SEQUENCE // 2)]
    df_test = sequence.iloc[(LEN_SEQUENCE // 2):]

    df_test_with_anomalies, true_anomalies = put_anomalies(df_test, params)

    return df_train, df_test_with_anomalies, true_anomalies


def save_pdf(pdf: PdfPages, df: pd.DataFrame, true_anomalies: pd.DataFrame, title: str) -> None:
    figure_anonymous = get_figure(df, true_anomalies=true_anomalies, title=title,
                                  orion_format=True)
    figure_anonymous.savefig(pdf, format='pdf')


def combine_data(df_train: pd.DataFrame, df_test: pd.DataFrame, true_anomalies: pd.DataFrame) -> dict:
    combined_dict = {'df_train': df_train.to_dict(),
                     'df_test': df_test.to_dict(),
                     'true_anomalies': true_anomalies.to_dict()}

    return combined_dict


def save_json(data: dict, filename: str) -> None:
    with open(filename, 'w') as file:
        json.dump(data, file)


def main() -> None:
    benchmark_path = get_path('benchmark')
    benchmark_data_path = get_path(benchmark_path, 'data')

    pdf_anonymous = PdfPages(f'{benchmark_path}/anonymous_plots.pdf')
    pdf_named = PdfPages(f'{benchmark_path}/named_plots.pdf')

    anomaly_generation_params = get_anomaly_generation_params()

    sequences_dict = {}
    n = 0
    data_files = get_data_files()

    for file in tqdm(data_files, desc='Data', leave=False, total=len(data_files)):
        benchmark_data_file_path = get_path(benchmark_data_path, file.name[:-4])
        sequences = get_random_sequences_from_data(read_df(file.path))

        for i, sequence in tqdm(enumerate(sequences), desc='Sequences', leave=False, total=len(sequences)):
            sequences_dict[f'{file.name}_{i + 1}'] = orion_format(sequence).to_dict()

            for test in range(NUM_TEST):
                df_train, df_test_with_anomalies, true_anomalies = get_data(sequence, anomaly_generation_params)

                save_pdf(pdf_anonymous, df_test_with_anomalies, true_anomalies, title=f'{n + 1}')
                save_pdf(pdf_named, df_test_with_anomalies, true_anomalies,
                         title=f'[{n + 1}] data: {file.name}, sequence: {i + 1}, test: {test + 1}')

                combined_dict = combine_data(df_train, df_test_with_anomalies, true_anomalies)

                save_json(combined_dict, filename=f'{benchmark_data_file_path}/{n + 1}.json')

                n += 1

        pdf_anonymous.close()
        pdf_named.close()

        save_json(sequences_dict, filename=f'{benchmark_path}/sequences.json')


if __name__ == '__main__':
    main()
