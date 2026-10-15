"""Release integrity and failure behavior for read-only hosted inference."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import joblib
from fastapi import HTTPException

from backend import release
from ml.artifacts import file_hash
from ml.release import build_release
from test_predictions import game
from test_retrain import dataset


def write_release(folder):
    bundle = {"schema_version": 2, "report": {"source_sha256": "a" * 64}}
    joblib.dump(bundle, folder / "models.joblib")
    manifest = {"schema_version": 1, "runtime": release.runtime_versions(),
                "model_sha256": file_hash(folder / "models.joblib"), "source_sha256": "a" * 64}
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return bundle, manifest


class HostingTests(unittest.TestCase):
    def tearDown(self):
        release.load_release.cache_clear()

    def test_verified_release_needs_no_training_files_or_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            bundle, _ = write_release(folder)
            with patch.object(Path, "write_text", side_effect=AssertionError("Runtime wrote a file")):
                self.assertEqual(release.read_release(folder), bundle)
            self.assertEqual({path.name for path in folder.iterdir()}, {"manifest.json", "models.joblib"})

    def test_corruption_and_runtime_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for corruption in ("bytes", "runtime", "source", "schema"):
                with self.subTest(corruption=corruption):
                    _, manifest = write_release(folder)
                    if corruption == "bytes":
                        (folder / "models.joblib").write_bytes(b"corrupt")
                    elif corruption == "runtime":
                        manifest["runtime"]["python"] = "0.0"
                    elif corruption == "source":
                        manifest["source_sha256"] = "b" * 64
                    else:
                        manifest["schema_version"] = 999
                    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        release.read_release(folder)

    def test_missing_release_returns_503_and_can_recover(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(release, "RELEASE", Path(directory)):
            with self.assertLogs(release.logger, level="ERROR"), self.assertRaises(HTTPException) as error:
                release.load_release()
            self.assertEqual(error.exception.status_code, 503)
            bundle, _ = write_release(Path(directory))
            self.assertEqual(release.load_release(), bundle)

    def test_invalid_data_never_starts_hosting_training(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            rows = game()
            rows[0]["date"] = "2026-01-01"
            source = dataset(folder / "source.csv", rows)
            with patch("ml.release.train_snapshot") as train:
                with self.assertRaises(ValueError):
                    build_release(source, folder / "release")
            train.assert_not_called()
            self.assertFalse((folder / "release").exists())

    def test_failed_training_does_not_replace_existing_release(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            output = folder / "release"
            output.mkdir()
            write_release(output)
            before = {path.name: path.read_bytes() for path in output.iterdir()}
            source = dataset(folder / "source.csv")
            with patch("ml.release.train_snapshot", side_effect=subprocess.CalledProcessError(1, "training")):
                with self.assertRaises(subprocess.CalledProcessError):
                    build_release(source, output)
            self.assertEqual(before, {path.name: path.read_bytes() for path in output.iterdir()})


if __name__ == "__main__":
    unittest.main()
