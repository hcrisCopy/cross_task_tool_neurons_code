from __future__ import annotations

from typing import Iterable, TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")


def progress(items: Iterable[T], **kwargs) -> Iterable[T]:
    return tqdm(items, dynamic_ncols=True, **kwargs)
