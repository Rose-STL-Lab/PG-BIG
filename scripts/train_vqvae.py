#!/usr/bin/env python3
"""Thin CLI wrapper — delegates to package main()."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> None:
    if "--help" in sys.argv or "-h" in sys.argv:
        import vqvae.config as option_vqvae
        option_vqvae.get_args_parser()
        return
    from vqvae.train import main as train_main
    train_main()


if __name__ == "__main__":
    main()
