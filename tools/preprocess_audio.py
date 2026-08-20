#!/usr/bin/env python3
"""Compatibility entry point for the unified H2S audio preprocessing CLI."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.preprocess.prepare_audio import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

