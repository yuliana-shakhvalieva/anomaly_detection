from src.common.constants import NUM_EPOCH, DATA_FREQUENCY, EPS
from src.model import constants

default_hyperparameters = {
    constants.ARIMA: {
        'mlstars.custom.timeseries_preprocessing.time_segments_aggregate#1': {
            'interval': DATA_FREQUENCY,
        },
        'custom.arima#1': {
            'trend': 't',
            'steps': 1,
        },
        'custom.find_anomalies#1': {
            'anomaly_padding': EPS
        },
    },

    constants.LINEAR_AE: {
        'keras.Sequential.DenseSeq2Seq#1': {
            'epochs': NUM_EPOCH,
            'verbose': True
        },
        'mlstars.custom.timeseries_preprocessing.time_segments_aggregate#1': {
            'interval': DATA_FREQUENCY,
        },
        'custom.find_anomalies#1': {
            'anomaly_padding': EPS
        },
    },

    constants.VAE: {
        'orion.primitives.vae.VAE#1': {
            'epochs': NUM_EPOCH,
            'verbose': True
        },
        'mlstars.custom.timeseries_preprocessing.time_segments_aggregate#1': {
            'interval': DATA_FREQUENCY,
        },
        'custom.find_anomalies#1': {
            'anomaly_padding': EPS
        },
    },

    constants.LSTM_AE: {
        'keras.Sequential.LSTMSeq2Seq#1': {
            'epochs': NUM_EPOCH,
            'verbose': True
        },
        'mlstars.custom.timeseries_preprocessing.time_segments_aggregate#1': {
            'interval': DATA_FREQUENCY,
        },
        'custom.find_anomalies#1': {
            'anomaly_padding': EPS
        },
    },

    constants.LSTM_THRESHOLD: {
        'keras.Sequential.LSTMTimeSeriesRegressor#1': {
            'epochs': NUM_EPOCH,
            'verbose': True
        },
        'mlstars.custom.timeseries_preprocessing.time_segments_aggregate#1': {
            'interval': DATA_FREQUENCY,
        },
        'custom.find_anomalies#1': {
            'anomaly_padding': EPS
        },
    },

    constants.TADGAN: {
        'orion.primitives.tadgan.TadGAN#1': {
            'epochs': NUM_EPOCH,
            'verbose': True,
        },
        'mlstars.custom.timeseries_preprocessing.time_segments_aggregate#1': {
            'interval': DATA_FREQUENCY,
        },
        'custom.find_anomalies#1': {
            'anomaly_padding': EPS
        },
    },

    constants.AER: {
        'orion.primitives.aer.AER#1': {
            'epochs': NUM_EPOCH,
            'verbose': True
        },
        'mlstars.custom.timeseries_preprocessing.time_segments_aggregate#1': {
            'interval': DATA_FREQUENCY,
        },
        'custom.find_anomalies#1': {
            'anomaly_padding': EPS
        },
    },

}
