"""Shared pytest fixtures.

Exposes lightweight fixtures so unit tests stay fast (no LLM calls). Heavier
fixtures (recorded LLM responses, sample documents) are added once those
modules land.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `scripts/` importable from the test suite (the package layout treats
# scripts/ as a top-level dir, not a package).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
