import multiprocessing

# Параметры для параллельных вычислений
CPU_COUNT = multiprocessing.cpu_count()                      # количество ядер процессора
CHUNK_SIZE = 10                                              # размер одного чанка

MULTIPROCESS_MODE = 'm'                                      # режим параллельных вычислений
SEQUENTIAL_MODE = 's'                                        # режим последовательных вычислений
