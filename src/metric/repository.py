from src.common.repository import ModelEntityRepository
from src.metric.abstract import Metric
from src.metric.utils.utils import get_mean


class MetricRepository(ModelEntityRepository):
    def __init__(self) -> None:
        super().__init__()

    def _register_model(self, model_name: str) -> None:
        self._repository[model_name] = dict()

    def __register_entity(self, model_name: str, metric_id: str) -> None:
        self._repository[model_name][metric_id] = list()

    def _save_entity(self, model_name: str, metrics: list[Metric]) -> None:
        for metric in metrics:
            if metric.id not in self._repository[model_name]:
                self.__register_entity(model_name, metric.id)
            self._repository[model_name][metric.id].append(metric.value)

    def fix_state(self) -> None:
        for model_name, metrics in self._repository.items():
            for metric_id, history_values in metrics.items():
                self._repository[model_name][metric_id] = get_mean(history_values)
