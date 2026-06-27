"""JSON I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["load_json", "save_json"]


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
