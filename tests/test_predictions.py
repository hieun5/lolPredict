"""Regression checks for data leakage, draft validity, and prediction input handling."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi import HTTPException
from pydantic import ValidationError

from backend.predictions import Draft, PredictionRequest, predict
from ml.build_features import ROLES, NUMERIC_FEATURES, add_history, model_features, pair_matches
from ml.train import split_by_date


def game(game_id="g1", date="2025-01-01T12:00:00Z", outcome=1):
    records = []
    for side, team, result, prefix in (("Blue", "A", outcome, "Blue"), ("Red", "B", 1 - outcome, "Red")):
        common = dict(gameid=game_id, date=date, league="LCK", patch="15.1",
                      side=side, teamname=team, teamid=team, result=result, gamelength=1800)
        records.append({**common, "position": "team", "champion": None})
        for role in ROLES:
            records.append({**common, "position": role, "champion": prefix + role})
    return records


class DataTests(unittest.TestCase):
    def test_pairing_and_draft_role_assignment(self):
        matches, quality = pair_matches(pd.DataFrame(game()))
        self.assertEqual(len(matches), 1)
        self.assertTrue(matches.iloc[0].draft_complete)
        self.assertEqual(matches.iloc[0].blue_jng, "Bluejng")
        self.assertEqual(quality["rejected_games"], 0)

    def test_missing_or_duplicate_champions_only_disable_draft(self):
        for champion in (None, "Redmid"):
            rows = game()
            rows[1]["champion"] = champion
            matches, quality = pair_matches(pd.DataFrame(rows))
            self.assertEqual(len(matches), 1)
            self.assertFalse(matches.iloc[0].draft_complete)
            self.assertEqual(quality["complete_drafts"], 0)

    def test_ambiguous_team_rows_are_rejected(self):
        rows = game() + game("g2")
        rows.append(rows[0].copy())
        matches, quality = pair_matches(pd.DataFrame(rows))
        self.assertEqual(matches.gameid.tolist(), ["g2"])
        self.assertEqual(quality["rejected_games"], 1)

    def test_conflicting_results_are_rejected(self):
        rows = game() + game("g2")
        rows[6]["result"] = 1
        matches, quality = pair_matches(pd.DataFrame(rows))
        self.assertEqual(matches.gameid.tolist(), ["g2"])
        self.assertEqual(quality["rejected_games"], 1)

    def test_own_and_future_outcomes_do_not_change_current_features(self):
        original, _ = pair_matches(pd.DataFrame(game() + game("g2", "2025-01-02T12:00:00Z")))
        changed = original.copy()
        changed.loc[1, "blue_win"] = 0
        first, _ = add_history(original)
        second, _ = add_history(changed)
        np.testing.assert_allclose(first[list(NUMERIC_FEATURES)], second[list(NUMERIC_FEATURES)])
        self.assertEqual(first.iloc[0].blue_games_before, 0)
        self.assertEqual(first.iloc[1].blue_games_before, 1)
        self.assertEqual(first.iloc[1].blue_wins_before, 1)

    def test_equal_timestamps_never_see_each_others_results(self):
        matches, _ = pair_matches(pd.DataFrame(game() + game("g2")))
        features, _ = add_history(matches)
        self.assertEqual(features.blue_games_before.tolist(), [0, 0])
        self.assertEqual(features.elo_diff.tolist(), [0, 0])

    def test_chronological_split_keeps_days_together(self):
        rows = [
            {"date": pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(hours=index * 12), "blue_win": index % 2}
            for index in range(100)
        ]
        train, validation, test = split_by_date(pd.DataFrame(rows))
        self.assertLess(train.date.max().floor("D"), validation.date.min().floor("D"))
        self.assertLess(validation.date.max().floor("D"), test.date.min().floor("D"))
        self.assertEqual(len(train) + len(validation) + len(test), 100)

    def test_only_allowlisted_features_reach_model(self):
        matches, _ = pair_matches(pd.DataFrame(game()))
        features, _ = add_history(matches)
        row = features.iloc[0].to_dict()
        row["kills"] = 99
        pre = model_features(row)
        draft = model_features(row, True)
        self.assertEqual(set(pre), {*NUMERIC_FEATURES, "league"})
        self.assertNotIn("blue_win", draft)
        self.assertNotIn("kills", draft)
        self.assertEqual(draft["role:top:Bluetop"], 1)
        self.assertEqual(draft["role:top:Redtop"], -1)


class InputTests(unittest.TestCase):
    def request(self, **overrides):
        return PredictionRequest(**{"league": "LCK", "blue_team_id": "A", "red_team_id": "B", **overrides})

    def draft(self, prefix):
        return Draft(**{role: prefix + role for role in ROLES})

    def test_pre_draft_needs_no_picks(self):
        self.assertIsNone(self.request().blue_draft)

    def test_same_team_rejected(self):
        with self.assertRaises(ValidationError):
            self.request(red_team_id="A")

    def test_partial_or_duplicate_draft_rejected(self):
        with self.assertRaises(ValidationError):
            self.request(blue_draft=self.draft("Blue"))
        with self.assertRaises(ValidationError):
            self.request(blue_draft=self.draft("Blue"), red_draft=self.draft("Blue"))
        with self.assertRaises(ValidationError):
            Draft(top="Gnar")

    def test_unknown_team_and_league_rejected(self):
        catalog = {"leagues": ["LCK"], "teams": [{"id": "A"}]}
        with patch("backend.predictions.load_bundle", return_value={"catalog": catalog}):
            for request in (self.request(), self.request(league="invalid")):
                with self.assertRaises(HTTPException) as error:
                    predict(request)
                self.assertEqual(error.exception.status_code, 422)


if __name__ == "__main__":
    unittest.main()

