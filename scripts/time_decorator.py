import time
from functools import wraps


# Define the timing decorator
def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(
            f"Method '{func.__name__}' executed in {end_time - start_time:.4f} seconds"
        )
        return result

    return wrapper


# Metaclass to apply the decorator
class TimingMeta(type):
    def __new__(cls, name, bases, dct):
        for attr_name, attr_value in dct.items():
            if callable(attr_value) and not attr_name.startswith("__"):
                # Handle staticmethod
                if isinstance(attr_value, staticmethod):
                    dct[attr_name] = staticmethod(timing_decorator(attr_value.__func__))
                # Handle classmethod
                elif isinstance(attr_value, classmethod):
                    dct[attr_name] = classmethod(timing_decorator(attr_value.__func__))
                else:
                    dct[attr_name] = timing_decorator(attr_value)
        return super().__new__(cls, name, bases, dct)
