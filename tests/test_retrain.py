"""Behavioral checks for local validation, failed training, activation, and hot reload."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from backend import predictions
from ml.artifacts import current_run, file_hash, model_path_for_run, write_json_atomic
from ml.build_features import build_dataset
from ml.retrain import activate_run, pipeline_fingerprint, retrain, snapshot_dataset
from ml.validate_data import validate_dataset
from test_predictions import game


class ConstantModel:
    def predict_proba(self, rows):
        return np.tile([0.4, 0.6], (len(rows), 1))


def dataset(path, rows=None):
    pd.DataFrame(rows if rows is not None else game()).to_csv(path, index=False)
    return path


def write_run(root, run_id, source):
    folder = model_path_for_run(run_id, root).parent
    folder.mkdir(parents=True, exist_ok=True)
    score = {"games": 1, "accuracy": .5, "log_loss": .7, "brier_score": .25}
    bundle = {
        "schema_version": 2,
        "catalog": {"teams": [{"id": "A"}], "champions": ["Gnar"], "leagues": ["LCK"]},
        "pre_model": ConstantModel(), "draft_model": ConstantModel(),
        "report": {"tag": run_id, "source_sha256": file_hash(source), "partitions": {"test": {"pre_draft": score, "post_draft": score}}},
    }
    joblib.dump(bundle, folder / "models.joblib")
    matches, _, _ = build_dataset(source)
    matches.to_csv(folder / "matches.csv", index=False)
    write_json_atomic(folder / "run.json", {"status": "complete", "model_sha256": file_hash(folder / "models.joblib")})


class ValidationTests(unittest.TestCase):
    def test_invalid_sources_are_rejected(self):
        mutations = {
            "duplicate role": lambda rows: rows.append(rows[0].copy()),
            "wrong year": lambda rows: rows[0].update(date="2026-01-01"),
            "invalid result": lambda rows: rows[0].update(result=2),
            "invalid patch": lambda rows: rows[0].update(patch="not-a-patch"),
            "conflicting duration": lambda rows: rows[0].update(gamelength=1900),
            "missing role": lambda rows: rows.pop(),
        }
        with tempfile.TemporaryDirectory() as folder:
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    rows = game()
                    mutate(rows)
                    with self.assertRaises(ValueError):
                        validate_dataset(dataset(Path(folder) / "source.csv", rows))

    def test_incomplete_draft_is_retained_and_reported(self):
        rows = game()
        rows[1]["champion"] = None
        with tempfile.TemporaryDirectory() as folder:
            report = validate_dataset(dataset(Path(folder) / "source.csv", rows))
        self.assertEqual(report["games"], 1)
        self.assertEqual(report["incomplete_drafts"], 1)

    def test_snapshot_is_immutable_and_deduplicated(self):
        with tempfile.TemporaryDirectory() as folder:
            source = dataset(Path(folder) / "source.csv")
            first, report = snapshot_dataset(source, Path(folder) / "snapshots")
            second, _ = snapshot_dataset(source, Path(folder) / "snapshots")
            self.assertEqual(first, second)
            source.write_text("changed", encoding="utf-8")
            self.assertEqual(file_hash(first), report["source_sha256"])

    def test_failed_validation_leaves_no_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "invalid.csv"
            source.write_text("wrong,columns\n1,2\n", encoding="utf-8")
            destination = Path(folder) / "snapshots"
            with self.assertRaises(ValueError):
                snapshot_dataset(source, destination)
            self.assertEqual(list(destination.iterdir()), [])


class RunTests(unittest.TestCase):
    def test_failed_training_preserves_previous_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "artifacts"
            source = dataset(Path(folder) / "source.csv")
            write_run(root, "prior", source)
            previous = activate_run("prior", root)
            with patch("ml.retrain.train_snapshot", side_effect=subprocess.CalledProcessError(1, "training")):
                with self.assertRaises(subprocess.CalledProcessError):
                    retrain(source, root=root, snapshots=Path(folder) / "snapshots")
            self.assertEqual(current_run(root), previous)
            statuses = [json.loads(path.read_text())["status"] for path in (root / "runs").glob("*/run.json")]
            self.assertIn("failed", statuses)

    def test_unchanged_dataset_and_code_skip_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "artifacts"
            source = dataset(Path(folder) / "source.csv")
            write_run(root, "prior", source)
            manifest_path = root / "runs/prior/run.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["fingerprint"] = pipeline_fingerprint(file_hash(source))
            write_json_atomic(manifest_path, manifest)
            activate_run("prior", root)
            with patch("ml.retrain.train_snapshot") as train:
                result = retrain(source, root=root, snapshots=Path(folder) / "snapshots")
            self.assertEqual(result["status"], "unchanged")
            train.assert_not_called()

    def test_successful_training_activates_verified_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "artifacts"
            source = dataset(Path(folder) / "source.csv")
            def train(snapshot, output, label):
                self.assertIsNone(current_run(root))
                write_run(root, output.name, snapshot)
            with patch("ml.retrain.train_snapshot", side_effect=train):
                result = retrain(source, root=root, snapshots=Path(folder) / "snapshots")
            self.assertEqual(result["status"], "activated")
            self.assertEqual(current_run(root)["run_id"], result["run_id"])

    def test_partial_or_modified_run_cannot_replace_working_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "artifacts"
            source = dataset(Path(folder) / "source.csv")
            write_run(root, "prior", source)
            previous = activate_run("prior", root)
            write_run(root, "candidate", source)
            model_path_for_run("candidate", root).write_bytes(b"corrupted")
            with self.assertRaises(ValueError):
                activate_run("candidate", root)
            write_json_atomic(root / "runs/candidate/run.json", {"status": "training"})
            with self.assertRaises(ValueError):
                activate_run("candidate", root)
            self.assertEqual(current_run(root), previous)

    def test_cached_api_observes_activation_and_rollback(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "artifacts"
            source = dataset(Path(folder) / "source.csv")
            write_run(root, "first", source)
            write_run(root, "second", source)
            predictions._load_bundle_at.cache_clear()
            with patch.object(predictions, "MODEL_PATH", root / "models.joblib"):
                activate_run("first", root)
                self.assertEqual(predictions.load_bundle()["report"]["tag"], "first")
                activate_run("second", root)
                self.assertEqual(predictions.load_bundle()["report"]["tag"], "second")
                activate_run("first", root)
                self.assertEqual(predictions.load_bundle()["report"]["tag"], "first")
            predictions._load_bundle_at.cache_clear()

    def test_run_paths_cannot_escape_artifact_directory(self):
        for run_id in ("../outside", "C:\\elsewhere", "bad/name", ""):
            with self.subTest(run_id=run_id), self.assertRaises(ValueError):
                model_path_for_run(run_id)


if __name__ == "__main__":
    unittest.main()
