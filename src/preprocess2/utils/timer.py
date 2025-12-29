import time
from contextlib import contextmanager


@contextmanager
def timer(name: str):
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"[TIMER] {name}: {elapsed:.2f}s")
