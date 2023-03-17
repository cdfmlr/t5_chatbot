import logging
from datetime import datetime


def cooldown(seconds: int):
    """Cooldown: a decorator to limit the frequency of function calls

    Args:
        seconds (int): seconds

    Returns:
        function: decorator
    """
    logging.debug(f"Cooldown: {seconds} seconds")

    def decorator(func):
        last_called = 0

        def wrapper(*args, **kwargs):
            if not kwargs.get('no_cooldown', False):  # no no_cooldown: do it
                nonlocal last_called
                now = datetime.now().timestamp()
                if now - last_called < seconds:
                    raise CooldownException(int(seconds - now + last_called))
                last_called = now
            return func(*args, **kwargs)

        return wrapper
    return decorator


class CooldownException(Exception):
    def __init__(self, seconds: int):
        super().__init__(f"Cooldown: {seconds} seconds")
