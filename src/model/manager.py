import pandas as pd

from src.common.constants import INPUT_DATA_LENGTH
from src.common.utils import concat_lists
from src.metric.abstract import AnomaliesCount, Metric
from src.metric.manager import MetricManager
from src.model.builder import ModelBuilder
from src.model.proxy import OrionModelProxy
from src.model.utils import get_y_true_and_y_pred
from src.utils.decorators.time import measure_time


class ModelManager:
    def __init__(self,
                 model_builder: ModelBuilder,
                 metric_manager: MetricManager) -> None:

        self.models = model_builder.build_models()
        self.metric_manager = metric_manager

    @measure_time('Fitting all models')
    def fit(self, df_train) -> None:
        for model in self.models.values():
            model.fit(df_train)

    def get_by_name(self, name: str) -> OrionModelProxy:
        return self.models[name]

    def __test_on_full_df_test(self,
                               model: OrionModelProxy,
                               df_test: pd.DataFrame,
                               true_anomalies: pd.DataFrame) -> (list[Metric], pd.DataFrame):

        pred_anomalies, time_detection_metrics = self.metric_manager.measure_over_invoke(model, df_test)
        y_true, y_pred = get_y_true_and_y_pred(df_test, true_anomalies, pred_anomalies)
        calculated_metrics = self.metric_manager.calculate(y_true, y_pred,
                                                           AnomaliesCount(true=true_anomalies.shape[0],
                                                                          pred=pred_anomalies.shape[0]))

        return concat_lists(calculated_metrics, time_detection_metrics), pred_anomalies

    @measure_time('Testing all models on full df_test')
    def test_on_full_df_test(self,
                             df_test: pd.DataFrame,
                             true_anomalies: pd.DataFrame) -> (dict[str, list[Metric]], dict[str, pd.DataFrame]):

        model_metrics, model_pred_anomalies = dict(), dict()

        for model_name, model in self.models.items():
            metrics, pred_anomalies = self.__test_on_full_df_test(model, df_test, true_anomalies)
            model_metrics[model_name] = metrics
            model_pred_anomalies[model_name] = pred_anomalies

        return model_metrics, model_pred_anomalies

    def __test(self,
               model: OrionModelProxy,
               window_size: int,
               df_train: pd.DataFrame,
               df_test: pd.DataFrame,
               true_anomalies: pd.DataFrame,
               make_fit: bool) -> (list[Metric], pd.DataFrame):

        result_time_metrics = []
        result_pred_anomalies = pd.DataFrame(columns=['start', 'end'])
        size = 200

        for j in range(0, len(df_test), INPUT_DATA_LENGTH):
            if j < size:
                train_part = df_train[-(size-j):]
                test_part = df_test[:j]
                history_df = pd.concat([train_part, test_part], axis=0)
            else:
                history_df = df_test[j - size:j]

            target_df = df_test[j: j + INPUT_DATA_LENGTH]
            df = pd.concat([history_df, target_df], axis=0)

            if not make_fit:
                pred_anomalies, time_metrics = self.metric_manager.measure_over_invoke(model, df)
            else:
                pred_anomalies, time_metrics = self.metric_manager.measure_over_invoke(model, df, history_df)

            result_time_metrics.extend(time_metrics)
            anomalies = list(pred_anomalies[['start', 'end']].itertuples(index=False))

            if len(anomalies) > 0:
                target_timestamps = target_df.timestamp.values

                for anomaly in anomalies:
                    start, end = anomaly
                    if start in target_timestamps and end in target_timestamps:
                        result_pred_anomalies.loc[result_pred_anomalies.shape[0]] = [start, end]

                    elif start in target_timestamps and end not in target_timestamps:
                        result_pred_anomalies.loc[result_pred_anomalies.shape[0]] = [start, target_timestamps[-1]]

                    elif start not in target_timestamps and end in target_timestamps:
                        result_pred_anomalies.loc[result_pred_anomalies.shape[0]] = [target_timestamps[0], end]

        y_true, y_pred = get_y_true_and_y_pred(df_test, true_anomalies, result_pred_anomalies)
        calculated_metrics = self.metric_manager.calculate(y_true, y_pred,
                                                           AnomaliesCount(true=true_anomalies.shape[0],
                                                                          pred=result_pred_anomalies.shape[0]))

        return concat_lists(calculated_metrics, result_time_metrics), result_pred_anomalies

    @measure_time('Testing all models')
    def test(self,
             window_size: int,
             df_train: pd.DataFrame,
             df_test: pd.DataFrame,
             true_anomalies: pd.DataFrame,
             make_fit: bool) -> (dict[str, list[Metric]], dict[str, pd.DataFrame]):

        model_metrics, model_pred_anomalies = dict(), dict()

        for model_name, model in self.models.items():
            metrics, pred_anomalies = self.__test(model, window_size, df_train, df_test, true_anomalies, make_fit)
            model_metrics[model_name] = metrics
            model_pred_anomalies[model_name] = pred_anomalies

        return model_metrics, model_pred_anomalies
