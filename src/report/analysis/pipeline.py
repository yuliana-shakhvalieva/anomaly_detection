import os

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import src.report.common.utils
from src.report.analysis import utils


def get_valid_data_paths(path: str) -> list[(str, str)]:
    valid_data_paths = []
    for data_name in os.listdir(path):
        if not data_name.startswith('.') and not data_name.endswith('.csv'):
            valid_data_paths.append((os.path.join(path, data_name), data_name))

    return valid_data_paths


def get_metrics_by_files(path: str) -> dict[str, dict[str, dict[str, float]]]:
    metrics_by_files = dict()

    for file_path, file_name in utils.get_valid_reports_filenames(path):
        metrics_by_files[file_name] = utils.get_metrics_from_file_by_models(file_path)

    return metrics_by_files


def get_metrics_by_models(path: str) -> dict[str, dict[str, dict[str, float]]]:
    metrics_by_files = dict()

    for file_path, file_name in utils.get_valid_reports_filenames(path):
        for model_name, metrics in utils.get_metrics_from_file_by_models(file_path).items():
            if model_name not in metrics_by_files.keys():
                metrics_by_files[model_name] = {file_name: metrics}
            else:
                metrics_by_files[model_name].update({file_name: metrics})

    return metrics_by_files


def save_plots(current_path: str,
               metrics_by_files: dict[str, dict[str, float]],
               df_best_files_by_metrics: pd.DataFrame,
               best_final_files: list[str]) -> None:
    file_names, metrics_for_plots = src.report.common.utils.get_info_for_plots(metrics_by_files)

    pdf = PdfPages(current_path + '/' + 'plots.pdf')

    for metric_name, metric_value in metrics_for_plots.items():
        fig = plt.figure(figsize=(13, 5))
        plt.plot(file_names, metric_value, zorder=2)
        plt.title(metric_name, size=15)
        plt.ylabel('value', size=11)
        plt.xlabel('file name', size=11)

        metric_row = df_best_files_by_metrics[df_best_files_by_metrics.metric == metric_name]

        if not metric_row.empty:
            best_value = metric_row.best_value.tolist()[0]
            best_files = metric_row.best_files.tolist()[0]

            for i, best_file in enumerate(best_files):
                plt.scatter(best_file, best_value, zorder=3, color='#1f77b4', label='_' * i + 'Best metric value')

        for j, best_final_file in enumerate(best_final_files):
            plt.scatter(best_final_file, metric_value[file_names.index(best_final_file)],
                        zorder=3, color='#d62728', label='_' * j + 'Final best file')

        plt.grid(zorder=1)
        plt.legend()

        fig.savefig(pdf, format='pdf')
        plt.close()

    pdf.close()


def add_best_values(df_result: pd.DataFrame, data_name: str, model_name: str, df_best_files: pd.DataFrame) -> list[str]:
    best_files = df_best_files.best_files.explode()

    final_best_files = best_files.mode()
    total_count = best_files.shape[0]
    value_counts = best_files.value_counts()

    mode_values, percents = [], []
    for final_best_file in final_best_files:
        mode_value = final_best_file[:-4]
        count_mode_value = value_counts.loc[final_best_file]
        percent = round(count_mode_value / total_count, 2)

        mode_values.append(mode_value)
        percents.append(percent)

    df_result.loc[df_result.shape[0]] = [data_name, model_name, mode_values, percents]

    return final_best_files
