"""Atomic selection of immutable local training runs."""

import hashlib
import json
import os
from pathlib import Path
import re
import uuid

from ml.build_features import ROOT

ARTIFACTS = ROOT / "artifacts"


def file_hash(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def model_path_for_run(run_id: str, root: Path = ARTIFACTS) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Invalid training run ID.")
    if run_id == "legacy":
        return root / "models.joblib"
    return root / "runs" / run_id / "models.joblib"


def current_run(root: Path = ARTIFACTS) -> dict | None:
    pointer = root / "current.json"
    if not pointer.exists():
        return None
    try:
        current = json.loads(pointer.read_text(encoding="utf-8"))
        if current["schema_version"] != 1 or not re.fullmatch(r"[a-f0-9]{64}", current["model_sha256"]):
            raise ValueError("Invalid active model metadata.")
        model_path_for_run(current["run_id"], root)
        return current
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("Invalid active model metadata.") from error
