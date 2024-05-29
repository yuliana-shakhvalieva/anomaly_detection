from typing import Any, Callable


def log_model(text: str = None) -> Callable:
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any or None:
            self = args[0]
            total_count = self.total_count
            i = self.id
            name = self.name

            print(f'\n[{i + 1}/{total_count}] {name}\n{text}')

            return func(*args, **kwargs)

        return wrapper

    return decorator


def log(text: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any or None:
            print(text)
            return func(*args, **kwargs)

        return wrapper

    return decorator
