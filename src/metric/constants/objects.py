from src.metric import measurers, calculators

METRIC_CALCULATORS = [calculators.ClassificationReportCalculator(),
                      calculators.IoUCalculator(),
                      calculators.FalsePositiveCalculator(),
                      calculators.NotDetectedCalculator(),
                      calculators.LateDetectionCalculator(),
                      calculators.GoodDetectionCalculator()]

MODEL_DETECT_TIME_MEASURER = measurers.ModelDetectTimeMeasurer()
ARIMA_MODEL_COUNT_DETECT_MEASURER = measurers.ArimaModelCountMeasurer(MODEL_DETECT_TIME_MEASURER)
SINGLE_ARIMA_DETECT_TIME_MEASURER = measurers.SingleArimaDetectTimeMeasurer(ARIMA_MODEL_COUNT_DETECT_MEASURER)

MODEL_FIT_DETECT_TIME_MEASURER = measurers.ModelFitDetectTimeMeasurer()
ARIMA_MODEL_COUNT_FIT_DETECT_MEASURER = measurers.ArimaModelCountMeasurer(MODEL_FIT_DETECT_TIME_MEASURER)
SINGLE_ARIMA_FIT_DETECT_TIME_MEASURER = measurers.SingleArimaDetectTimeMeasurer(ARIMA_MODEL_COUNT_FIT_DETECT_MEASURER)
