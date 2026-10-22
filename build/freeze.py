"""Write every filter/transform constant to docs/thresholds.json with a SHA-256 over the content.

Usage: python build/freeze.py   (run after the last audit round, before any model is trained)
"""
import hashlib
import json
import subprocess
from pathlib import Path

from sen_bridge import constants as C

vals = {k: v for k, v in vars(C).items() if k.isupper()}
body = json.dumps(vals, sort_keys=True, indent=2)
out = {"constants": vals, "sha256": hashlib.sha256(body.encode()).hexdigest(),
       "code_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
       "source_commit": "97a91733db962222e1e6b40a836845ddbbb16b86", "seed": 0}
Path("docs").mkdir(exist_ok=True)
Path("docs/thresholds.json").write_text(json.dumps(out, indent=2) + "\n")
print(out["sha256"])
