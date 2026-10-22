# Copyright 2026 Nikhil Raghavendra
# SPDX-License-Identifier: Apache-2.0
"""Write every filter and transform constant to ``docs/thresholds.json``.

The snapshot carries a SHA-256 of the constants. Run it after the last audit
round and before any model is trained.

Usage::

    python build/freeze.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import]  (fixed git call)
from pathlib import Path

from sen_bridge_build import constants

#: Hugging Face commit of SSL4EO-S12 v1.1 that the build reads.
SOURCE_COMMIT = "97a91733db962222e1e6b40a836845ddbbb16b86"
_SEED = 0


def git_head() -> str:
    """Return the commit hash of the current checkout.

    Returns:
        The full hash, or an empty string if git is unavailable.
    """
    git = shutil.which("git")
    if git is None:
        return ""
    # Fixed arguments, no user input.
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [git, "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip()


def main() -> None:
    """Write the thresholds snapshot and print its digest."""
    values = {k: v for k, v in vars(constants).items() if k.isupper()}
    body = json.dumps(values, sort_keys=True, indent=2)
    snapshot = {
        "constants": values,
        "sha256": hashlib.sha256(body.encode()).hexdigest(),
        "code_commit": git_head(),
        "source_commit": SOURCE_COMMIT,
        "seed": _SEED,
    }
    Path("docs").mkdir(exist_ok=True)
    Path("docs/thresholds.json").write_text(
        json.dumps(snapshot, indent=2) + "\n", encoding="utf-8"
    )
    print(snapshot["sha256"])


if __name__ == "__main__":
    main()
