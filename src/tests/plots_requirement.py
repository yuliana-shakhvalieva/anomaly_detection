from dataclasses import dataclass

import pandas as pd


@dataclass
class PlotsRequirements:
    df_test: pd.DataFrame
    true_anomalies: pd.DataFrame
    model_pred_anomalies: dict[str, pd.DataFrame]
