from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence, TypeVar

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    tqdm = None  # type: ignore[assignment]

T = TypeVar("T")


def progress(items: Iterable[T], **kwargs) -> Iterable[T]:
    if tqdm is not None:
        return tqdm(items, dynamic_ncols=True, **kwargs)
    return _plain_progress(items, **kwargs)


def progress_chunks(items: Sequence[T], chunk_size: int, **kwargs: Any) -> Iterator[Sequence[T]]:
    chunk_size = max(1, int(chunk_size))
    total = len(items)
    if tqdm is not None:
        bar_kwargs = dict(kwargs)
        bar_kwargs.setdefault("unit", "task")
        bar_kwargs.setdefault("dynamic_ncols", True)
        with tqdm(total=total, **bar_kwargs) as bar:
            for start in range(0, total, chunk_size):
                chunk = items[start : start + chunk_size]
                yield chunk
                bar.update(len(chunk))
        return

    desc = kwargs.get("desc") or "tasks"
    done = 0
    for start in range(0, total, chunk_size):
        chunk = items[start : start + chunk_size]
        yield chunk
        done += len(chunk)
        print(f"{desc}: {done}/{total}")


class ProgressTracker:
    def __init__(self, path: str | Path | None, *, total: int) -> None:
        self.path = Path(path) if path else None
        self.total = int(total)
        self.done = 0
        self.write()

    def update(self, value: int) -> None:
        self.done = min(self.total, self.done + int(value))
        self.write()

    def write(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"done": self.done, "total": self.total, "updated_at": time.time()}
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)


def evaluate_batched_with_task_progress(
    w2t_utils: Any,
    tasks: Sequence[dict[str, Any]],
    model: Any,
    *,
    batch_size: int,
    desc: str,
    tracker: ProgressTracker | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for chunk in progress_chunks(tasks, batch_size, desc=desc, unit="task"):
        outputs.extend(w2t_utils.evaluate_batched(list(chunk), model, **kwargs))
        if tracker is not None:
            tracker.update(len(chunk))
    return outputs


def _plain_progress(items: Iterable[T], **kwargs) -> Iterator[T]:
    desc = kwargs.get("desc") or "progress"
    total = kwargs.get("total")
    if total is None and hasattr(items, "__len__"):
        total = len(items)  # type: ignore[arg-type]
    for idx, item in enumerate(items, start=1):
        print(f"{desc}: {idx}/{total if total is not None else '?'}")
        yield item
