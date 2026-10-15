"""Read the immutable serving bundle generated during the hosting build."""

from functools import lru_cache
from importlib.metadata import version
import json
import logging
from pathlib import Path
import sys

import joblib
from fastapi import HTTPException

from ml.artifacts import file_hash

RELEASE = Path(__file__).resolve().parents[1] / "release"
PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "joblib")
logger = logging.getLogger(__name__)


def runtime_versions() -> dict:
    return {"python": f"{sys.version_info.major}.{sys.version_info.minor}",
            **{name: version(name) for name in PACKAGES}}


def read_release(folder: Path) -> dict:
    """Verify before unpickling; only load bundles created by our own build."""
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("runtime") != runtime_versions():
        raise ValueError("Release format or Python/package versions do not match this runtime.")
    model = folder / "models.joblib"
    if file_hash(model) != manifest.get("model_sha256"):
        raise ValueError("Release model checksum does not match its manifest.")
    bundle = joblib.load(model)
    if bundle.get("schema_version") != 2 or bundle["report"]["source_sha256"] != manifest["source_sha256"]:
        raise ValueError("Release model metadata does not match its manifest.")
    return bundle


@lru_cache(maxsize=1)
def load_release() -> dict:
    try:
        return read_release(RELEASE)
    except Exception as error:
        logger.exception("Could not initialize the hosted model release")
        raise HTTPException(503, "The prediction model is unavailable. Please try again later.") from error
