from src.common.repository import ModelEntityRepository
from src.tests.utils import PlotsRequirements
from src.visualization.abstract import Visualizer
from src.visualization.utils import get_figure


class PlotsRepository(ModelEntityRepository):
    def __init__(self, visualizers: list[Visualizer]) -> None:
        super().__init__()
        self.visualizers = visualizers

    def save(self, plots_requirements: PlotsRequirements) -> None:
        df_test = plots_requirements.df_test
        true_anomalies = plots_requirements.true_anomalies
        model_pred_anomalies = plots_requirements.model_pred_anomalies

        for model_name, pred_anomalies in model_pred_anomalies.items():
            figure = get_figure(df_test, true_anomalies=true_anomalies, pred_anomalies=pred_anomalies, title=model_name)
            self._save(model_name, figure)

    def visualize(self) -> None:
        for visualizer in filter(lambda vis: vis.is_enabled(), self.visualizers):
            for model_name, figures in self._repository.items():
                visualizer.visualize(model_name, figures)
