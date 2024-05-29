from src.metric.constants import names

METRICS_DESCRIPTIONS = {names.IOU: 'Intersection over Union',
                        names.GOOD_DETECTION: 'Процент вовремя обнаруженных аномалий',
                        names.FALSE_POSITIVES: 'Процент ложных срабатываний',
                        names.LATE_DETECTION: 'Процент поздно обнаруженных аномалий',
                        names.NOT_DETECTED: 'Процент не обнаруженных аномалий',
                        names.CLASSIFICATION_REPORT: '',
                        names.DETECT_TIME: 'Общее время детекции (сек)',
                        names.FIT_DETECT_TIME: 'Общее время детекции (сек)',
                        names.ARIMA_DETECT_TIME: 'Время работы одной ARIMA модели (сек)',
                        names.ARIMA_MODELS_COUNT: 'Количество построенных ARIMA моделей за одну итерацию'}
