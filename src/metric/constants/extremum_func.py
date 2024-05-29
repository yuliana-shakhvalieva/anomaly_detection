from src.metric.constants import names
from src.metric.constants.descriptions import METRICS_DESCRIPTIONS

BARE_METRICS_EXTREMUM_FUNC = {names.ACCURACY: max,
                              names.MACRO_AVG_PRECISION: max,
                              names.MACRO_AVG_RECALL: max,
                              names.MACRO_AVG_F1_SCORE: max,
                              names.WEIGHTED_AVG_PRECISION: max,
                              names.WEIGHTED_AVG_RECALL: max,
                              names.WEIGHTED_AVG_F1_SCORE: max,
                              names.IOU: max,
                              names.FALSE_POSITIVES: min,
                              names.NOT_DETECTED: min,
                              names.LATE_DETECTION: min,
                              names.GOOD_DETECTION: max,
                              names.DETECT_TIME: min}

METRICS_EXTREMUM_FUNC = {METRICS_DESCRIPTIONS[metric]: func
                         for metric, func in BARE_METRICS_EXTREMUM_FUNC.items()
                         if metric in METRICS_DESCRIPTIONS.keys()}
METRICS_EXTREMUM_FUNC.update(BARE_METRICS_EXTREMUM_FUNC)
