import numpy as np
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima import model
from tqdm.auto import trange


class ARIMA:
    def __init__(self, trend: str, steps: int):
        self.trend = trend
        self.steps = steps

    def predict(self, X: np.ndarray) -> np.ndarray:
        arima_results = list()
        dimensions = len(X.shape)

        if dimensions > 2:
            raise ValueError("Only 1D o 2D arrays are supported")

        if dimensions == 1 or X.shape[1] == 1:
            X = np.expand_dims(X, axis=0)

        arima = auto_arima(y=X[0], seasonal=False, trace=False, max_p=5, max_q=5)
        order = arima.order

        num_sequences = len(X)
        for sequence in trange(num_sequences, desc=f'ARIMA {order}', leave=False):
            try:
                arima = model.ARIMA(X[sequence], order=order, trend=self.trend)
                arima_fit = arima.fit()
            except Exception as ex:
                print(ex)
                arima = model.ARIMA(X[sequence], order=(0, 0, 0), trend=self.trend)
                arima_fit = arima.fit()
            arima_results.extend(arima_fit.forecast(self.steps))

        arima_results = np.asarray(arima_results)

        if dimensions == 1:
            arima_results = arima_results[0]

        return arima_results
