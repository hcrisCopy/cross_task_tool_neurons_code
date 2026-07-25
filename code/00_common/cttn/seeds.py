from __future__ import annotations

import hashlib
from typing import Any

ALLOWED_SEEDS = (2026, 42, 123456)
DEFAULT_SEED = 2026


def seed_arg_kwargs() -> dict[str, Any]:
    return {"type": int, "default": DEFAULT_SEED, "choices": ALLOWED_SEEDS}


def derive_allowed_seed(base_seed: int, *parts: object) -> int:
    if int(base_seed) not in ALLOWED_SEEDS:
        raise ValueError(f"Seed must be one of {ALLOWED_SEEDS}, got {base_seed}")
    payload = "|".join([str(int(base_seed)), *[str(part) for part in parts]])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return ALLOWED_SEEDS[digest[0] % len(ALLOWED_SEEDS)]
