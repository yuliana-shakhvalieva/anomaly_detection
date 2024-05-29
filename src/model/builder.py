from src.constants.custom_types import CUSTOM_HYPERPARAMETERS_TYPE, ORION_HYPERPARAMETERS_TYPE
from src.model.constants import DEFAULT_MODELS
from src.model.proxy import OrionModelProxy
from src.model.utils import build_proxy, get_hyperparameters


class ModelBuilder:
    def __init__(self, *, model_names: str or list[str] = None, hyperparameters: CUSTOM_HYPERPARAMETERS_TYPE = None) -> None:
        self.model_names = model_names
        self.hyperparameters = hyperparameters

    def build_models(self) -> dict[str, OrionModelProxy]:
        model_names = self.__get_model_names()
        total_count = len(model_names)
        hyperparameters = self.__get_hyperparameters(model_names)
        return {name: build_proxy(i, total_count, name, hyperparameter)
                for i, (name, hyperparameter) in enumerate(zip(model_names, hyperparameters))}

    def __get_model_names(self) -> list[str]:
        if self.model_names is None:
            return DEFAULT_MODELS
        elif isinstance(self.model_names, str):
            return [self.model_names]
        else:
            return self.model_names

    def __get_hyperparameters(self, model_names: list[str]) -> list[ORION_HYPERPARAMETERS_TYPE]:
        return [get_hyperparameters(self.hyperparameters, name) for name in model_names]
