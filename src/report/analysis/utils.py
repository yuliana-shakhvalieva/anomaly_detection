import csv
import os
import re

from src.model.constants import DEFAULT_MODELS
from src.report.analysis.constants import BINARY_COLUMNS, BEST_FILES_NAME


def get_columns_result_df(name: str) -> list[str]:
    return ['data', 'model', f'best_{name}', 'percent']


def dfs(used: list[str], valid_reports_filenames: list[(str, str)], path: str):
    used.append(path)

    for filename in os.listdir(path):
        if (BEST_FILES_NAME not in filename and
                filename.endswith('.csv') and
                not filename.startswith('.')):
            valid_reports_filenames.append((os.path.join(path, filename), filename))

        elif (os.path.join(path, filename) not in used and
              not filename.endswith('.pdf') and
              not filename.endswith('.csv') and
              not filename.startswith('.')):
            dfs(used, valid_reports_filenames, os.path.join(path, filename))


def get_valid_reports_filenames(path: str) -> list[(str, str)]:
    used = []
    valid_reports_filenames = []

    dfs(used, valid_reports_filenames, path)

    # valid_reports_filenames.sort(key=lambda x: float(re.findall(r'\d+', x[1])[0]))

    return valid_reports_filenames


def get_metrics_from_one_model(relevant_info: list[[list[str]]]) -> dict[str, float]:
    metrics = dict()
    for items in relevant_info:
        if len(items) == 5:
            metric_name = items[0].replace(' ', '_')

            if metric_name == 'accuracy':
                metrics['accuracy'] = float(items[3])
            else:
                for i, column in enumerate(BINARY_COLUMNS):
                    full_metric_name = metric_name + '_' + str(column)
                    metrics[full_metric_name] = float(items[i + 1])

        elif len(items) == 2 and items != ['', '']:
            metric_name, metric_value = items
            metrics[metric_name] = float(metric_value)

    return metrics


def get_metrics_from_file_by_models(file_path: str) -> dict[str, dict[str, float]]:
    metrics_by_models = dict()

    with open(file_path) as f:
        csv_reader = csv.reader(f)
        info = list(csv_reader)

        for i, row in enumerate(info):
            if len(row) > 0 and row[0] in DEFAULT_MODELS:
                metrics_by_models[row[0]] = get_metrics_from_one_model(info[i + 4:i + 14])

    return metrics_by_models
