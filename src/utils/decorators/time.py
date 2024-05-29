from typing import Callable, Any

from src.common.utils import measure_seconds


def measure_time(text: str = None) -> Callable:
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any or None:
            result, spent_time = measure_seconds(func, *args, **kwargs)
            round_time = round(spent_time / 60, 2)

            if text is not None:
                print(f'\nTotal time spent on {text.lower()} (min): {round_time}\n')
            else:
                print(f'Time spent (min): {round_time}')

            return result

        return wrapper

    return decorator
