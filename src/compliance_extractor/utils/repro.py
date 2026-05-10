"""Reproducibility helpers: seed setting and run-id generation."""
from __future__ import annotations

import hashlib
import os
import random
from datetime import datetime, timezone


def set_global_seed(seed: int) -> None:
    """Seed Python and numpy if available. Torch is not touched (optional dep)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def run_id(*, prefix: str = "run") -> str:
    """Stable, sortable run id: `{prefix}_{utc-iso-no-colons}_{8-char-rand-hex}`."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rand = hashlib.sha256(os.urandom(16)).hexdigest()[:8]
    return f"{prefix}_{ts}_{rand}"
