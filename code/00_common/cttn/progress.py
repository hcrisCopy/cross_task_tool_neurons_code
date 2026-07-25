from __future__ import annotations

from typing import Iterable, Iterator, TypeVar

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    tqdm = None  # type: ignore[assignment]

T = TypeVar("T")


def progress(items: Iterable[T], **kwargs) -> Iterable[T]:
    if tqdm is not None:
        return tqdm(items, dynamic_ncols=True, **kwargs)
    return _plain_progress(items, **kwargs)


def _plain_progress(items: Iterable[T], **kwargs) -> Iterator[T]:
    desc = kwargs.get("desc") or "progress"
    total = kwargs.get("total")
    if total is None and hasattr(items, "__len__"):
        total = len(items)  # type: ignore[arg-type]
    for idx, item in enumerate(items, start=1):
        print(f"{desc}: {idx}/{total if total is not None else '?'}")
        yield item
