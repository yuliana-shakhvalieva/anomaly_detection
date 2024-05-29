from typing import Any

from src.constants.custom_types import CLASSIFICATION_REPORT_TYPE


def get_pretty_title(text: str) -> str:
    return ('\033[1m' + text + '\033[0m').center(55)


def pretty_print(text: str, values: Any) -> None:
    print(f'\n\n{get_pretty_title(text)}', values, sep='\n\n')


def print_classification_report_dict(classification_report_dict: CLASSIFICATION_REPORT_TYPE) -> None:
    labels = ['0', '1', 'accuracy', 'macro avg', 'weighted avg']
    headers = ['precision', 'recall', 'f1-score', 'support']

    longest_last_line_heading = 'weighted avg'
    name_width = max(len(cn) for cn in labels)
    width = max(name_width, len(longest_last_line_heading), 2)
    head_fmt = '{:>{width}s} ' + ' {:>9}' * len(headers)

    classification_report_str = head_fmt.format('', *headers, width=width)
    classification_report_str += '\n\n'

    rows = []
    for label in labels:
        if label == 'accuracy':
            rows.append((label, *classification_report_dict[label]))
        else:
            rows.append((label, *classification_report_dict[label].values()))

    row_fmt = '{:>{width}s} ' + ' {:>9.{digits}f}' * 3 + ' {:>9}\n'
    for row in rows:
        if row[0] == 'accuracy':
            classification_report_str += '\n'
            row_fmt_accuracy = (
                    '{:>{width}s} '
                    + ' {:>9.{digits}}' * 2
                    + ' {:>9.{digits}f}'
                    + ' {:>9}\n'
            )
            classification_report_str += row_fmt_accuracy.format(
                "accuracy", "", "", *row[1:], width=width, digits=2
            )
        else:
            classification_report_str += row_fmt.format(*row, width=width, digits=2)
    classification_report_str += "\n"

    print(classification_report_str)

#
# def print_best_models(best_models, title):
#     metrics_name = MetricsName()
#
#     content = f'{metrics_name.accuracy}:\t\t\t\t\t{best_models.accuracy_best}\n'
#     content += f'{metrics_name.macro_avg_precision}:\t\t\t\t{best_models.macro_avg_precision_best}\n'
#     content += f'{metrics_name.macro_avg_recall}:\t\t\t\t{best_models.macro_avg_recall_best}\n'
#     content += f'{metrics_name.macro_avg_f1_score}:\t\t\t\t{best_models.macro_avg_f1_score_best}\n'
#
#     content += f'{metrics_name.weighted_avg_precision}:\t\t\t\t{best_models.weighted_avg_precision_best}\n'
#     content += f'{metrics_name.weighted_avg_recall}:\t\t\t\t{best_models.weighted_avg_recall_best}\n'
#     content += f'{metrics_name.weighted_avg_f1_score}:\t\t\t\t{best_models.weighted_avg_f1_score_best}\n'
#
#     content += f'{metrics_name.iou}:\t\t\t{best_models.iou_best}\n'
#     content += f'{metrics_name.false_positives}:\t\t\t{best_models.false_positives_best}\n'
#     content += f'{metrics_name.not_detected}:\t\t{best_models.not_detected_best}\n'
#     content += f'{metrics_name.late_detection}:\t\t{best_models.late_detection_best}\n'
#     content += f'{metrics_name.good_detection}:\t\t{best_models.good_detection_best}'
#
#     pretty_print(title, content)

# todo print_best_models
