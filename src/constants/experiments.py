from src.common.constants import NUM_SEQUENCES, LEN_SEQUENCE, NUM_TEST, INPUT_DATA_LENGTH

WINDOW_SIZE = 'window_size'
STEPS = 'steps'
ANOMALY_SIZE = 'anomaly_size'

START_EXPERIMENTS_DESCRIPTIONS = {
    WINDOW_SIZE: 'Подбор window_size_rolling_window_sequences на реальных данных из файла',
    STEPS: 'Подбор steps с лучшим значением window_size_rolling_window_sequences на реальных данных из файла',
    ANOMALY_SIZE: 'Подбор window_size_find_anomalies с лучшим значением window_size_rolling_window_sequences на '
                  'реальных данных из файла'
}

START_EXPERIMENTS_DESCRIPTIONS_BENCH = {
    WINDOW_SIZE: 'Подбор window_size_rolling_window_sequences на данных из бенчмарка, созданных на основе файла',
    STEPS: 'Подбор steps с лучшим значением window_size_rolling_window_sequences на данных из бенчмарка, созданных на '
           'основе файла',
    ANOMALY_SIZE: 'Подбор window_size_find_anomalies с лучшим значением window_size_rolling_window_sequences на '
                  'данных из бенчмарка, созданных на основе файла'
}

END_EXPERIMENT_DESCRIPTION = (f'взяты {NUM_SEQUENCES} случайных кусочков длины {LEN_SEQUENCE} (половина в test), '
                              f'на каждом кусочке было проведено {NUM_TEST} тестов, первая часть без аномалий, '
                              f'расчет метрик происходит без первой части')

END_EXPERIMENTS_DESCRIPTIONS_BENCH = (f'тестовые данные разбиты на кусочки длиной {INPUT_DATA_LENGTH}, к каждому '
                                      f'кусочку слева были добавлены исторических данные длины '
                                      f'window_size_rolling_window_sequences')

FIT_ON_HISTORY_DATA = ', обучение происходило на исторических данных'
FIT_DETECT = ', fit_detect'
SINGLE_FIT = ', обучение один раз на train'
