from src.model import constants

PIPELINES = {constants.ARIMA: 'src/orion_applications/pipelines/arima.json',
             constants.LINEAR_AE: 'src/orion_applications/pipelines/dense_autoencoder.json',
             constants.VAE: 'src/orion_applications/pipelines/vae.json',
             constants.LSTM_AE: 'src/orion_applications/pipelines/lstm_autoencoder.json',
             constants.LSTM_THRESHOLD: 'src/orion_applications/pipelines/lstm_dynamic_threshold.json',
             constants.TADGAN: 'src/orion_applications/pipelines/tadgan.json',
             constants.AER: 'src/orion_applications/pipelines/aer.json'}
