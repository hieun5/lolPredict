"""Vercel build hook; run from the repository root with Python 3.14 and Node 24."""

import json
import os
import shutil
import subprocess
import sys

from ml.build_features import ROOT, default_source
from ml.release import build_release


def main():
    if sys.version_info[:2] != (3, 14):
        raise RuntimeError("The hosting build requires Python 3.14 (see .python-version).")
    npm = shutil.which("npm")
    if npm is None:
        raise RuntimeError("Install Node.js 24 before building the hosted site.")
    # Hosting always uses /api on its own origin, including when a developer has
    # a custom local VITE_API_BASE_URL in their shell.
    environment = {**os.environ, "VITE_API_BASE_URL": "/api"}
    subprocess.run([npm, "ci"], cwd=ROOT / "frontend", check=True, env=environment)
    subprocess.run([npm, "run", "build"], cwd=ROOT / "frontend", check=True, env=environment)
    manifest = build_release(default_source(), ROOT / "release")
    print(json.dumps({"hosting_release": manifest}, indent=2))


if __name__ == "__main__":
    main()
