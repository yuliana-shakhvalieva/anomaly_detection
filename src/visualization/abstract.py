from typing import NoReturn

from matplotlib.figure import Figure


class Visualizer:
    def __init__(self, mode: bool) -> None:
        self.enabled = mode

    def is_enabled(self) -> bool:
        return self.enabled

    def visualize(self, model_name: str, figures: list[Figure]) -> NoReturn:
        raise NotImplementedError
