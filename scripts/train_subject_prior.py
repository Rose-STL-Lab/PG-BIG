#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> None:
    from profile.train_prior import main as train_main
    train_main()


if __name__ == "__main__":
    main()
