import pandas as pd
from tqdm.auto import tqdm

import src.constants.experiments as experiments
import src.report.analysis.pipeline as pipeline
from src.report.analysis.constants import BEST_FILES_NAME, RESULT_CSV_FILE_NAME
from src.report.analysis.utils import get_columns_result_df
from src.report.common.utils import get_df_with_best_keys_by_metrics, save_plots_to_pdf


def main() -> None:
    experiment_name = experiments.WINDOW_SIZE

    reports_path = '/media/yuliana/DATA/studies/diplom_results/real_data/9. choose method/' + experiment_name

    df_result = pd.DataFrame(columns=get_columns_result_df(experiment_name))
    valid_data_paths = pipeline.get_valid_data_paths(reports_path)

    for current_path, data_name in tqdm(valid_data_paths, desc='Data', leave=True, total=len(valid_data_paths)):
        metrics_by_models = pipeline.get_metrics_by_models(current_path)

        for model_name, metrics_by_files in tqdm(metrics_by_models.items(), desc='Models', leave=False, total=len(metrics_by_models)):
            df_best_files_by_metrics = get_df_with_best_keys_by_metrics(metrics_by_files,
                                                                        pd.DataFrame(
                                                                            columns=['metric', 'best_files',
                                                                                     'best_value']))

            best_final_files = pipeline.add_best_values(df_result, data_name, model_name, df_best_files_by_metrics)
            save_plots_to_pdf(current_path + '/' + f'{model_name}_metrics.pdf', metrics_by_files, df_best_files_by_metrics,
                              best_final_files, 'file')

            best_values = '_'.join(map(lambda x: x[:-4], best_final_files))
            df_best_files_by_metrics.to_csv(current_path + '/' + f'{model_name}_{BEST_FILES_NAME + best_values}.csv', index=False)

    df_result.to_csv(reports_path + '/' + RESULT_CSV_FILE_NAME, index=False)


if __name__ == '__main__':
    main()
