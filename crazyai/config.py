"""Paths and defaults."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
DOMAINS_DIR = DATA_DIR / "domains"
PROJECT_DIR = PACKAGE_DIR.parent
ARCHIVE_DIR = Path(os.environ.get("CRAZYAI_ARCHIVE", PROJECT_DIR / "archive"))

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_EFFORT = "high"
DEFAULT_MAX_TOKENS = 16000
DEFAULT_MAX_TOOL_TURNS = 24

OPERATORS = ["INVERT", "REMOVE", "EXTRAPOLATE", "TRANSPOSE", "COMPOSE", "QUANTIFY", "SUBSTITUTE"]
GENERATORS = ["story", "formula", "plan", "statmodel", "questions", "debate"]
