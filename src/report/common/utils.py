from typing import Callable

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from src.common.utils import add_to_dict_list
from src.metric.constants.descriptions import METRICS_DESCRIPTIONS
from src.metric.constants.extremum_func import METRICS_EXTREMUM_FUNC


def get_best_keys_by_metrics(metrics_by_keys: dict[str, dict[str, float]],
                             metric_name: str,
                             extremum_func: Callable) -> (list[str], float):
    temp_list = [(key, metrics[metric_name]) for key, metrics in metrics_by_keys.items()]
    extremum_value = extremum_func(temp_list, key=lambda x: x[1])[1]
    best_keys = [key for (key, metric_value) in temp_list if metric_value == extremum_value]
    return best_keys, extremum_value


def get_df_with_best_keys_by_metrics(metrics_by_keys: dict[str, dict[str, float]], df: pd.DataFrame) -> pd.DataFrame:
    current_metrics = list(list(metrics_by_keys.values())[0])
    for metric_name, extremum_func in METRICS_EXTREMUM_FUNC.items():
        if metric_name in current_metrics:
            best_files_names, best_value = get_best_keys_by_metrics(metrics_by_keys, metric_name, extremum_func)
            full_name = METRICS_DESCRIPTIONS[metric_name] if metric_name in METRICS_DESCRIPTIONS.keys() else metric_name
            df.loc[df.shape[0]] = [full_name, best_files_names, best_value]
    return df


def get_info_for_plots(metrics_by_keys: dict[str, dict[str, float]]) -> [list[str], dict[str, list[float]]]:
    metrics_for_plots = dict()
    for key, metrics in metrics_by_keys.items():
        for metric_name, metric_value in metrics.items():
            full_name = METRICS_DESCRIPTIONS[metric_name] if metric_name in METRICS_DESCRIPTIONS.keys() else metric_name
            add_to_dict_list(metrics_for_plots, [full_name, 'x'], key)
            add_to_dict_list(metrics_for_plots, [full_name, 'y'], metric_value)

    return metrics_for_plots

def split(handles_labels, plot_type):
    import matplotlib

    types = dict(plot=matplotlib.lines.Line2D,
                 scatter=matplotlib.collections.PathCollection,)

    try: plot_type = types[plot_type]
    except KeyError: raise ValueError('Invalid plot type.')

    return zip(*((h, l) for h, l in zip(*handles_labels) if type(h) is plot_type))


def save_important_plots_to_pdf(pdf_name: str,
                                metrics_by_keys: dict[str, dict[str, float]],
                                df_best_keys_by_metrics: pd.DataFrame,
                                best_final_keys: list[str],
                                key_name: str) -> None:
    metrics_for_plots = get_info_for_plots(metrics_by_keys)

    business_metrics = ['Общее время детекции (сек)']
    intersection_metrics = ['Процент вовремя обнаруженных аномалий', 'Процент ложных срабатываний', 'Intersection over Union']

    pdf = PdfPages(pdf_name)

    for global_metrics, title in zip([business_metrics, intersection_metrics],
                                     ['Общее время детекции (сек)', 'Метрики качества']):
        fig = plt.figure(figsize=(13, 5))
        for metric_name in global_metrics:
            info_for_plot = metrics_for_plots[metric_name]
            key_names = info_for_plot['x']
            metric_values = info_for_plot['y']

            if len(key_names) > 1:
                if len(global_metrics) == 1:
                    plt.plot(key_names, metric_values, zorder=2)
                else:
                    plt.plot(key_names, metric_values, zorder=2, label=metric_name)
                plt.title(title, size=15)
                plt.ylabel('value', size=11)
                plt.xlabel(f'{key_name} name', size=11)

                for j, best_final_key in enumerate(best_final_keys):
                    if best_final_key in key_names:
                        idx = key_names.index(best_final_key)
                        plt.scatter(best_final_key, metric_values[idx], zorder=3, color='#d62728')

        plt.grid(zorder=1)

        handles_labels = plt.gca().get_legend_handles_labels()

        l0 = plt.gca().legend(*split(handles_labels, 'plot'), loc='best')
        plt.gca().add_artist(l0)
        l1 = plt.gca().legend(*split(handles_labels, 'scatter'), loc='best')

        fig.savefig(pdf, format='pdf')
        plt.close()

    pdf.close()


def save_plots_to_pdf(pdf_name: str,
                      metrics_by_keys: dict[str, dict[str, float]],
                      df_best_keys_by_metrics: pd.DataFrame,
                      best_final_keys: list[str],
                      key_name: str) -> None:
    metrics_for_plots = get_info_for_plots(metrics_by_keys)

    pdf = PdfPages(pdf_name)

    for metric_name, info_for_plot in metrics_for_plots.items():
        key_names = info_for_plot['x']
        metric_values = info_for_plot['y']

        if len(key_names) > 1:
            fig = plt.figure(figsize=(13, 5))
            plt.plot(key_names, metric_values, zorder=2)
            plt.title(metric_name, size=15)
            plt.ylabel('value', size=11)
            plt.xlabel(f'{key_name} name', size=11)

            metric_row = df_best_keys_by_metrics[df_best_keys_by_metrics.metric == metric_name]

            if not metric_row.empty:
                best_keys = metric_row.iloc[:, 1].tolist()[0]
                best_value = metric_row.iloc[:, 2].tolist()[0]

                for i, best_key in enumerate(best_keys):
                    plt.scatter(best_key, best_value, zorder=3, color='#1f77b4', label='_' * i + 'Best metric value')

            for j, best_final_key in enumerate(best_final_keys):
                if best_final_key in key_names:
                    idx = key_names.index(best_final_key)
                    plt.scatter(best_final_key, metric_values[idx],
                                zorder=3, color='#d62728', label='_' * j + f'Final best {key_name}')

            plt.grid(zorder=1)
            plt.legend()

            fig.savefig(pdf, format='pdf')
            plt.close()

    pdf.close()
