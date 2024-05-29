from src.common import constants
from src.constants import generation

WORDS_IN_DESCRIPTION_PER_ROW = 8

TITLE = f'Средние показатели на {constants.NUM_TEST * constants.NUM_SEQUENCES} тестовых данных'

TITLE_DATA_GENERATION_PARAMS = ['Параметры генерации исходных данных']

DATA_GENERATION_PARAMS = [['Частотность данных в секундах', 'data_frequency', constants.DATA_FREQUENCY],
                          ['Количество дней для генерации', 'count_days', generation.COUNT_DAYS],
                          ['Мат ожидание для норм распределения, параметр шума', 'mean', generation.MEAN],
                          ['Масштаб данных', 'scale', generation.SCALE],
                          ['Константа, добавляемая к данным', 'add', generation.ADD]]

TITLE_ANOMALY_GENERATION_PARAMS = ['Параметры генерации аномалий на тестовых данных']

ANOMALY_GENERATION_PARAMS = [['Вероятность аномалии', 'p_anomaly'],
                             ['Допустимые длины аномалий', 'possible_anomaly_length'],
                             ['Начиная с какого номера начинать попытки добавления аномалий', 'start_from']]

TITLE_OTHER_GENERAL_PARAMS = ['Прочие параметры']

OTHER_GENERAL_PARAMS = [['Длительность промежутка, в котором должна быть обнаружена аномалия', 'eps', constants.EPS]]

TITLE_BEST_MODELS_BY_METRICS = [['Название метрики', 'Список лучших моделей', 'Лучшее значение метрики']]
