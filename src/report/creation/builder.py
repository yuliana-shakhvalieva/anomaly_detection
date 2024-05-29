from typing import Any

from src.metric.repository import MetricRepository
from src.model.manager import ModelManager
from src.report.common.utils import save_plots_to_pdf
from src.report.creation import pipeline


class ReportBuilder:
    def __init__(self,
                 model_manager: ModelManager,
                 metric_repository: MetricRepository,
                 data_generation: bool = False) -> None:

        self.model_manager = model_manager
        self.metric_repository = metric_repository
        self.data_generation = data_generation

    def __save_base_report(self,
                           report_file_name: str,
                           description: str,
                           anomaly_generation_params: dict[str, Any] or None) -> None:
        pipeline.start_report(report_file_name, description)

        if self.data_generation:
            pipeline.add_data_generation_params(report_file_name)

        if anomaly_generation_params is not None:
            pipeline.add_anomaly_generation_params(report_file_name, anomaly_generation_params)

        pipeline.add_other_general_params(report_file_name)

        for model_name, mean_metrics in self.metric_repository.get().items():
            model = self.model_manager.get_by_name(model_name)
            pipeline.add_model_results(report_file_name, model, mean_metrics)
            pipeline.add_model_params(report_file_name, model)
            pipeline.add_orion_processing_params(report_file_name, model)

    def save(self, report_file_name: str, description: str, anomaly_generation_params: dict[str, Any] or None) -> None:
        self.__save_base_report(report_file_name, description, anomaly_generation_params)

        metrics_by_models, df_best_models_names = pipeline.add_best_models_by_metrics(report_file_name,
                                                                                      self.metric_repository.get())
        best_final_model = pipeline.get_best_final_model(df_best_models_names)
        save_plots_to_pdf(report_file_name[:-4] + '_best_models.pdf',
                          metrics_by_models,
                          df_best_models_names,
                          best_final_model,
                          'model')  # todo: чтобы лучшие модели сохранялись в отдельный файл + рефакторинг Andrey
