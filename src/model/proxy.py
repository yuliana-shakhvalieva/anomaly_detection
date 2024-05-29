from typing import Any

import pandas as pd
from orion import Orion

from src.constants.custom_types import ORION_HYPERPARAMETERS_TYPE
from src.orion_applications.constants import PIPELINES
from src.report.creation.utils import get_window_size_rolling_window_sequences
from src.utils.decorators.loggers import log_model
from src.utils.decorators.time import measure_time


class OrionModelProxy:
    def __init__(self,
                 id: int,
                 total_count: int,
                 name: str,
                 orion_hyperparameters: ORION_HYPERPARAMETERS_TYPE,
                 model_hyperparameters: dict[str, Any] or None = None) -> None:

        self.id = id
        self.total_count = total_count
        self.name = name

        if model_hyperparameters is None:
            self.model_hyperparameters = dict()
        else:
            self.model_hyperparameters = model_hyperparameters

        self.model = Orion(pipeline=PIPELINES[name],
                           hyperparameters=orion_hyperparameters)

    @log_model('Fitting...')
    @measure_time()
    def fit(self, df_train: pd.DataFrame) -> None:
        self.model.fit(df_train)

    @log_model('Detecting...')
    @measure_time('Detecting')
    def detect(self, df_test: pd.DataFrame) -> pd.DataFrame:
        return self.model.detect(df_test)

    @log_model('Fit_detecting...')
    @measure_time('Fit_detecting')
    def fit_detect(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.model.fit_detect(df)


class ArimaProxy(OrionModelProxy):
    def __init__(self, i, total_count, name, orion_hyperparameters) -> None:
        self.df_before_test = None
        super().__init__(i, total_count, name, orion_hyperparameters, orion_hyperparameters['custom.arima#1'])

    # def fit(self, df_train: pd.DataFrame) -> None:
    #     window_size = get_window_size_rolling_window_sequences(self.model)
    #     self.df_before_test = df_train.iloc[-window_size:]
    #     super().fit(df_train)
    #
    # def detect(self, df_test: pd.DataFrame) -> pd.DataFrame:
    #     modified_df_test = pd.concat([self.df_before_test, df_test], axis=0)
    #     return super().detect(modified_df_test)
