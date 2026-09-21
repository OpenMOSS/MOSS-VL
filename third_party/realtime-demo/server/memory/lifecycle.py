"""Per-session admission barrier, shared by async callers and worker threads."""
from contextlib import contextmanager
from functools import wraps
import inspect
import threading


class SessionLifetime:
    def __init__(self):
        self._condition = threading.Condition()
        self._closing = False
        self._active = 0

    @property
    def closing(self):
        with self._condition:
            return self._closing

    @contextmanager
    def operation(self):
        with self._condition:
            admitted = not self._closing
            if admitted:
                self._active += 1
        try:
            yield admitted
        finally:
            if admitted:
                with self._condition:
                    self._active -= 1
                    self._condition.notify_all()

    def seal(self):
        with self._condition:
            self._closing = True

    def wait(self):
        """Call off the event loop, after seal() and async task cancellation."""
        with self._condition:
            self._condition.wait_for(lambda: self._active == 0)


def while_open(default=None):
    """Account for the whole operation, including awaits, without serializing it."""
    def decorate(method):
        if inspect.iscoroutinefunction(method):
            @wraps(method)
            async def wrapped(self, *args, **kwargs):
                with self.lifetime.operation() as admitted:
                    if not admitted:
                        return default
                    return await method(self, *args, **kwargs)
        else:
            @wraps(method)
            def wrapped(self, *args, **kwargs):
                with self.lifetime.operation() as admitted:
                    if not admitted:
                        return default
                    return method(self, *args, **kwargs)
        return wrapped
    return decorate
