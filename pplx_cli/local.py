"""Scoped local-only policy shared by generation and retrieval components."""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator


_local_only: ContextVar[bool] = ContextVar("pplx_local_only", default=False)


def local_only_enabled() -> bool:
    """Whether the active operation forbids remote inference and downloads."""
    return _local_only.get()


@contextmanager
def local_only_mode(enabled: bool = True) -> Iterator[None]:
    """Enable local policy for this operation without relaxing an outer scope."""
    token = _local_only.set(enabled or _local_only.get())
    try:
        yield
    finally:
        _local_only.reset(token)
