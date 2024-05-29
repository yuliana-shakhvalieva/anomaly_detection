from src.constants.counting_way import CHUNK_SIZE, MULTIPROCESS_MODE, SEQUENTIAL_MODE


# todo: убрать my и vk, как разработка закончится
def get_main_path(args: list[str]) -> str:
    if len(args) == 1:
        raise ValueError('Please enter a path for storing results (or use "my"/"vk").')
    elif args[1] == 'my':
        path = '/media/yuliana/DATA/studies/vk/results/real_data/'
    elif args[1] == 'vk':
        path = '/data1/y.shakhvalieva/results/real_data/'
    else:
        path = args[1]

    return path


def get_command_line_params(args: list[str]) -> (str, int):
    counting_way = MULTIPROCESS_MODE
    chunk_size = CHUNK_SIZE

    if len(args) == 2:
        return counting_way, chunk_size
    elif args[2] == SEQUENTIAL_MODE:
        counting_way = SEQUENTIAL_MODE
    else:
        try:
            chunk_size = int(args[2])
        except ValueError:
            raise ValueError(
                f'Unknown argument: {args[2]}. Please use "{SEQUENTIAL_MODE}" in case of sequential operating mode, '
                f'or enter an int chunk_size for multiprocess mode.')

    return counting_way, chunk_size
