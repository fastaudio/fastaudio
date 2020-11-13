import os
from functools import wraps


def skip_if_ci(func):
    @wraps(func)
    def _inner(alt_func=lambda x: None):
        if "CI" in os.environ and os.environ["CI"] == "true":
            return alt_func()
        return func()

    return _inner
