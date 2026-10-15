"""Validate and retrain from the local 2025 dataset, preserving the last working run."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import joblib
import numpy as np
import pandas as pd

from ml.artifacts import ARTIFACTS, current_run, file_hash, model_path_for_run, write_json_atomic
from ml.build_features import ROOT, default_source, model_features
from ml.validate_data import validate_dataset


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def pipeline_fingerprint(source_hash: str) -> str:
    digest = hashlib.sha256(source_hash.encode())
    for path in sorted((ROOT / "ml").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    digest.update((ROOT / "requirements.txt").read_bytes())
    return digest.hexdigest()


def snapshot_dataset(source: Path, folder: Path) -> tuple[Path, dict]:
    folder.mkdir(parents=True, exist_ok=True)
    candidate = folder / f".candidate-{uuid.uuid4().hex}.csv"
    try:
        shutil.copyfile(source, candidate)
        validation = validate_dataset(candidate)
        snapshot = folder / f"{validation['source_sha256']}.csv"
        if snapshot.exists():
            if file_hash(snapshot) != validation["source_sha256"]:
                raise ValueError("An existing dataset snapshot is corrupted.")
        else:
            candidate.replace(snapshot)
        return snapshot, validation
    finally:
        candidate.unlink(missing_ok=True)


def validate_model(model_path: Path, matches_path: Path | None = None) -> dict:
    bundle = joblib.load(model_path)
    if bundle.get("schema_version") != 2:
        raise ValueError("Unsupported model artifact schema.")
    catalog, report = bundle["catalog"], bundle["report"]
    if not catalog["teams"] or not catalog["champions"] or not catalog["leagues"]:
        raise ValueError("The trained catalog is empty.")
    for phase in ("pre_draft", "post_draft"):
        score = report["partitions"]["test"][phase]
        if score["games"] <= 0 or not all(np.isfinite(score[name]) for name in ("accuracy", "log_loss", "brier_score")):
            raise ValueError("The training run has invalid evaluation results.")
    if matches_path is not None:
        matches = pd.read_csv(matches_path, dtype={"patch": "string"})
        for draft, model in ((False, bundle["pre_model"]), (True, bundle["draft_model"])):
            cohort = matches[matches["draft_complete"]] if draft else matches
            if cohort.empty:
                raise ValueError("The run has no eligible model verification games.")
            probabilities = np.asarray(model.predict_proba([model_features(cohort.iloc[0].to_dict(), draft)]))
            if (
                probabilities.shape != (1, 2) or not np.isfinite(probabilities).all()
                or (probabilities < 0).any() or (probabilities > 1).any()
                or not np.isclose(probabilities.sum(), 1)
            ):
                raise ValueError("The trained model returned invalid probabilities.")
    return bundle


def activate_run(run_id: str, root: Path = ARTIFACTS) -> dict:
    model_path = model_path_for_run(run_id, root)
    matches_path = None
    if run_id != "legacy":
        manifest = json.loads((model_path.parent / "run.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise ValueError("Only a completed, verified run can be activated.")
        if file_hash(model_path) != manifest["model_sha256"]:
            raise ValueError("Model checksum does not match the completed run.")
        matches_path = model_path.parent / "matches.csv"
    bundle = validate_model(model_path, matches_path)
    pointer = {
        "schema_version": 1, "run_id": run_id, "activated_at": now(),
        "model_sha256": file_hash(model_path),
        "source_sha256": bundle["report"]["source_sha256"],
    }
    write_json_atomic(root / "current.json", pointer)
    return pointer


def train_snapshot(snapshot: Path, folder: Path, source_name: str) -> None:
    subprocess.run([
        sys.executable, "-m", "ml.train", "--source", str(snapshot),
        "--source-label", source_name, "--output", str(folder),
        "--processed-output", str(folder / "matches.csv"),
    ], cwd=ROOT, check=True)


def retrain(source: Path, *, force: bool = False, root: Path = ARTIFACTS, snapshots: Path | None = None) -> dict:
    snapshot, validation = snapshot_dataset(source, snapshots or ROOT / "data/snapshots")
    fingerprint = pipeline_fingerprint(validation["source_sha256"])
    current = current_run(root)
    if current is not None and current["run_id"] != "legacy" and not force:
        path = model_path_for_run(current["run_id"], root)
        manifest = json.loads((path.parent / "run.json").read_text(encoding="utf-8"))
        if manifest.get("fingerprint") == fingerprint and manifest.get("status") == "complete":
            if file_hash(path) != current["model_sha256"]:
                raise ValueError("The active model checksum is invalid.")
            return {"status": "unchanged", "run_id": current["run_id"], "validation": validation}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]
    folder = model_path_for_run(run_id, root).parent
    folder.mkdir(parents=True)
    manifest = {
        "run_id": run_id, "status": "training", "started_at": now(), "fingerprint": fingerprint,
        "source_name": source.name, "snapshot": str(snapshot.resolve()), "validation": validation,
    }
    write_json_atomic(folder / "run.json", manifest)
    try:
        train_snapshot(snapshot, folder, source.name)
        model_path = folder / "models.joblib"
        bundle = validate_model(model_path, folder / "matches.csv")
        if bundle["report"]["source_sha256"] != validation["source_sha256"]:
            raise ValueError("The model was not trained from the validated snapshot.")
        manifest.update({"status": "complete", "finished_at": now(), "model_sha256": file_hash(model_path)})
        write_json_atomic(folder / "run.json", manifest)
        activate_run(run_id, root)
    except Exception as error:
        manifest.update({"status": "failed", "finished_at": now(), "error": str(error)})
        write_json_atomic(folder / "run.json", manifest)
        raise
    return {"status": "activated", "run_id": run_id, "validation": validation}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Local 2025 CSV; defaults to the existing dataset")
    parser.add_argument("--force", action="store_true", help="Retrain even when source and pipeline are unchanged")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate-only", action="store_true", help="Validate without snapshots, training, or activation")
    modes.add_argument("--activate", metavar="RUN_ID", help="Reactivate a verified run, or 'legacy' for the original model")
    modes.add_argument("--status", action="store_true", help="Show the active model run")
    args = parser.parse_args()
    try:
        if args.status:
            result = current_run() or {"run_id": "legacy", "model": str(ARTIFACTS / "models.joblib")}
        elif args.activate:
            result = activate_run(args.activate)
        elif args.validate_only:
            result = {"status": "validated", "validation": validate_dataset(args.source or default_source())}
        else:
            result = retrain(args.source or default_source(), force=args.force)
        print(json.dumps(result, indent=2))
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Retraining failed: {error}\nThe active model selection was not changed.\n")


if __name__ == "__main__":
    main()
