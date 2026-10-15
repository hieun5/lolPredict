"""Check outcome availability, patch history, partition isolation, and metric edge cases."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from ml.build_features import (
    DRAFT_NUMERIC_FEATURES, NUMERIC_FEATURES, add_history, build_dataset,
    pair_matches, patch_features,
)
from ml.train import chronological_parts, metrics, probability_bins
from test_predictions import game


class ReliabilityTests(unittest.TestCase):
    def test_in_progress_games_do_not_update_team_or_champion_history(self):
        rows = game() + game("g2", "2025-01-01T12:15:00Z") + game("g3", "2025-01-01T12:31:00Z")
        paired, _ = pair_matches(pd.DataFrame(rows))
        features, _ = add_history(paired)
        self.assertEqual(features.blue_games_before.tolist(), [0, 0, 1])
        self.assertEqual(features.champion_patch_coverage.tolist(), [0, 0, 1])
        changed = paired.copy()
        changed.loc[0, "blue_win"] = 0
        changed_features, _ = add_history(changed)
        columns = [*NUMERIC_FEATURES, *DRAFT_NUMERIC_FEATURES]
        np.testing.assert_allclose(features.loc[1, columns].astype(float), changed_features.loc[1, columns].astype(float))

    def test_patch_change_resets_patch_exposure(self):
        rows = game() + game("g2", "2025-01-02T12:00:00Z")
        for row in rows[12:]:
            row["patch"] = "15.02"
        paired, _ = pair_matches(pd.DataFrame(rows))
        features, _ = add_history(paired)
        self.assertEqual(features.iloc[1].blue_games_before, 1)
        self.assertEqual(features.iloc[1].team_patch_coverage, 0)
        self.assertEqual(features.iloc[1].champion_patch_coverage, 0)
        # A new patch still has conservative, smoothed longer-term priors.
        self.assertGreater(features.iloc[1].team_patch_rate_diff, 0)

    def test_patch_strings_survive_csv_loading(self):
        rows = game() + game("g2", "2025-01-02T12:00:00Z")
        for row in rows[:12]:
            row["patch"] = "15.01"
        for row in rows[12:]:
            row["patch"] = "15.10"
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.csv"
            pd.DataFrame(rows).to_csv(source, index=False)
            matches, _, _ = build_dataset(source)
        self.assertEqual(matches.patch.tolist(), ["15.01", "15.10"])

    def test_serving_patch_lookup_matches_training_features(self):
        rows = game() + game("g2", "2025-01-02T12:00:00Z")
        paired, _ = pair_matches(pd.DataFrame(rows))
        features, _ = add_history(paired)
        _, store = add_history(paired.iloc[:1])
        row = paired.iloc[1].to_dict()
        serving = patch_features(store, row["blue_id"], row["red_id"], row["patch"], row)
        for name, value in serving.items():
            self.assertAlmostEqual(features.iloc[1][name], value)

    def test_partition_boundary_labels_are_available_before_next_window(self):
        rows = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=100, freq="12h", tz="UTC"),
            "blue_win": [index % 2 for index in range(100)],
        })
        rows["result_available_at"] = rows["date"] + pd.Timedelta(days=2)
        parts = chronological_parts(rows)
        for earlier, later in (("train", "calibration"), ("calibration", "validation"), ("validation", "test")):
            self.assertLess(parts[earlier].result_available_at.max(), parts[later].date.min())
            self.assertTrue(set(parts[earlier].index).isdisjoint(parts[later].index))

    def test_auc_is_measured_and_undefined_for_single_class(self):
        self.assertEqual(metrics([0, 1, 0, 1], [.1, .9, .2, .8])["roc_auc"], 1)
        self.assertIsNone(metrics([1, 1], [.6, .8])["roc_auc"])

    def test_calibration_bins_include_zero_and_one(self):
        bins = probability_bins([0, 1], [0, 1])
        self.assertEqual(sum(item["games"] for item in bins), 2)
        self.assertEqual(metrics([0, 1], [0, 1])["ece"], 0)


if __name__ == "__main__":
    unittest.main()
