from typing import Any

from src.constants.custom_types import PLOT_REPOSITORY_TYPE, METRIC_REPOSITORY_TYPE


class ModelEntityRepository:
    def __init__(self) -> None:
        self._repository: METRIC_REPOSITORY_TYPE or PLOT_REPOSITORY_TYPE = dict()

    def _register_model(self, model_name: str) -> None:
        self._repository[model_name] = list()

    def _save_entity(self, model_name: str, entity: Any) -> None:
        self._repository[model_name].append(entity)

    def _save(self, model_name: str, entity: Any) -> None:
        if model_name not in self._repository.keys():
            self._register_model(model_name)
        self._save_entity(model_name, entity)

    def save(self, model_entity: dict[str, Any]) -> None:
        for model_name, entity in model_entity.items():
            self._save(model_name, entity)

    def get(self) -> METRIC_REPOSITORY_TYPE or PLOT_REPOSITORY_TYPE:
        return self._repository
