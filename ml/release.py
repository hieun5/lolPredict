"""Build a standalone hosting model from the repository's existing 2025 data."""

import json
from pathlib import Path
import shutil
import tempfile

from backend.release import read_release, runtime_versions
from ml.artifacts import file_hash, write_json_atomic
from ml.build_features import ROOT, default_source
from ml.retrain import train_snapshot, validate_model
from ml.validate_data import validate_dataset


def build_release(source: Path, output: Path) -> dict:
    validation = validate_dataset(source)
    cache = ROOT / ".cache"
    cache.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hosting-", dir=cache) as folder:
        staging = Path(folder)
        train_snapshot(source, staging, source.name)
        model = staging / "models.joblib"
        bundle = validate_model(model, staging / "matches.csv")
        if bundle["report"]["source_sha256"] != validation["source_sha256"]:
            raise ValueError("The hosting model does not match the validated dataset.")
        manifest = {
            "schema_version": 1, "model_sha256": file_hash(model),
            "source_sha256": validation["source_sha256"], "runtime": runtime_versions(),
            "generated_at": bundle["report"]["generated_at"],
            "code_sha256": bundle["report"]["code_sha256"],
            "games": bundle["catalog"]["games"], "data_end": bundle["catalog"]["data_end"],
            "models": {"pre_draft": bundle["pre_model"].name, "post_draft": bundle["draft_model"].name},
        }
        write_json_atomic(staging / "manifest.json", manifest)
        read_release(staging)
        output.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(model, output / "models.joblib")
        write_json_atomic(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    print(json.dumps(build_release(default_source(), ROOT / "release"), indent=2))
