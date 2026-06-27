"""Central path defaults for PG-BIG.

Layout (default)::

    <repo>/datasets/183_athletes/
        Kinematic_Data/
        Participants Info/
        retargeted/          # per-subject .b3d files
"""

from __future__ import annotations

import os
from pathlib import Path

ATHLETES_183_SUBDIR = "183_athletes"
RETARGETED_SUBDIR = "retargeted"

__all__ = [
    "ATHLETES_183_DIR",
    "ATHLETES_183_RAW_DIR",
    "ATHLETES_183_RETARGETED_DIR",
    "ATHLETES_183_SUBDIR",
    "RETARGETED_SUBDIR",
    "default_athletes_root",
    "default_datasets_dir",
    "get_kinematic_directory",
    "get_participants_info_directory",
    "repo_root",
    "resolve_data_root",
    "resolve_repo_path",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_datasets_dir() -> Path:
    return repo_root() / "datasets"


def default_athletes_root() -> Path:
    explicit = os.environ.get("ATHLETES_DATA_ROOT", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    return default_datasets_dir() / ATHLETES_183_SUBDIR


def resolve_repo_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (repo_root() / p).resolve()


def resolve_data_root(path: str | None = None) -> str:
    if path is None or str(path).strip() == "":
        return str(default_athletes_root())
    return str(resolve_repo_path(path))


_athletes_root = default_athletes_root()
ATHLETES_183_DIR = str(_athletes_root)
ATHLETES_183_RAW_DIR = ATHLETES_183_DIR
ATHLETES_183_RETARGETED_DIR = str(_athletes_root / RETARGETED_SUBDIR)


def get_kinematic_directory(base_directory: str | Path) -> str:
    """Return the directory containing subject ID folders."""
    base = str(base_directory)
    candidates = [
        os.path.join(base, "Kinematic_Data", "Kinematic_Data"),
        os.path.join(base, "Kinematic_Data"),
    ]
    for path in candidates:
        if os.path.isdir(path) and any(
            d.isdigit()
            for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d))
        ):
            return path
    return os.path.join(base, "Kinematic_Data")


def get_participants_info_directory(base_directory: str | Path) -> str:
    """Return the directory containing participant metadata spreadsheets."""
    base = str(base_directory)
    candidates = [
        os.path.join(base, "Participants Info", "Participants Info"),
        os.path.join(base, "Participants Info"),
    ]
    for path in candidates:
        if os.path.isdir(path) and (
            os.path.exists(os.path.join(path, "Subject Log.xlsx"))
            or os.path.exists(os.path.join(path, "Sampling Frequency.xlsx"))
        ):
            return path
    return os.path.join(base, "Participants Info")
