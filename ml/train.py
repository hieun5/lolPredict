"""Reproducible chronological model comparison, calibration, evaluation, and serving refit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score, roc_curve

from ml.build_features import ROOT, ROLES, SIDES, HistoryStore, build_dataset, default_source, model_features
from ml.modeling import MatchModel

SEED = 42
MODEL_KINDS = ("logistic_regression", "random_forest")


def chronological_parts(matches: pd.DataFrame) -> dict[str, pd.DataFrame]:
    days = matches["date"].dt.floor("D")
    boundaries = [days.iloc[int(len(matches) * fraction)] for fraction in (0.60, 0.70, 0.85)]
    parts = {
        "train": matches[days < boundaries[0]],
        "calibration": matches[(days >= boundaries[0]) & (days < boundaries[1])],
        "validation": matches[(days >= boundaries[1]) & (days < boundaries[2])],
        "test": matches[days >= boundaries[2]],
    }
    # A label from a late game on the previous day may not exist at the next window's start.
    if "result_available_at" in matches:
        for earlier, later in (("train", "calibration"), ("calibration", "validation"), ("validation", "test")):
            cutoff = parts[later]["date"].min()
            parts[earlier] = parts[earlier][parts[earlier]["result_available_at"] < cutoff]
    if any(part.empty or part["blue_win"].nunique() != 2 for part in parts.values()):
        raise ValueError("Need four chronological partitions, each with both outcomes.")
    return parts


def split_by_date(matches: pd.DataFrame):
    parts = chronological_parts(matches)
    return pd.concat([parts["train"], parts["calibration"]]), parts["validation"], parts["test"]


def feature_rows(matches: pd.DataFrame, draft: bool) -> list[dict]:
    return [model_features(row, draft) for row in matches.to_dict("records")]


def make_estimator(kind: str, draft: bool):
    if kind == "logistic_regression":
        return LogisticRegression(C=0.05 if draft else 1.0, max_iter=1000, random_state=SEED)
    if kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=15,
            max_features=0.5, random_state=SEED, n_jobs=2,
        )
    raise ValueError(f"Unknown model kind: {kind}")


def fit_candidates(train: pd.DataFrame, calibration: pd.DataFrame, draft: bool) -> dict[str, MatchModel]:
    vectorizer = DictVectorizer()
    x_train = vectorizer.fit_transform(feature_rows(train, draft))
    x_calibration = vectorizer.transform(feature_rows(calibration, draft))
    candidates = {}
    for kind in MODEL_KINDS:
        estimator = make_estimator(kind, draft)
        estimator.fit(x_train, train["blue_win"])
        candidates[kind] = MatchModel(vectorizer, estimator, kind, draft, False)
        calibrated = CalibratedClassifierCV(FrozenEstimator(estimator), method="sigmoid")
        calibrated.fit(x_calibration, calibration["blue_win"])
        tag = f"{kind}_calibrated"
        candidates[tag] = MatchModel(vectorizer, calibrated, tag, draft, True)
    return candidates


def probability_bins(outcomes, probabilities) -> list[dict]:
    outcomes, probabilities = np.asarray(outcomes), np.asarray(probabilities)
    indexes = np.minimum((probabilities * 10).astype(int), 9)
    bins = []
    for index in range(10):
        mask = indexes == index
        if mask.any():
            bins.append({
                "lower": index / 10, "upper": (index + 1) / 10,
                "games": int(mask.sum()), "predicted": float(probabilities[mask].mean()),
                "observed": float(outcomes[mask].mean()),
            })
    return bins


def metrics(outcomes, probabilities) -> dict:
    outcomes, probabilities = np.asarray(outcomes), np.asarray(probabilities)
    bins = probability_bins(outcomes, probabilities)
    return {
        "games": len(outcomes),
        "accuracy": float(accuracy_score(outcomes, probabilities >= 0.5)),
        "roc_auc": float(roc_auc_score(outcomes, probabilities)) if np.unique(outcomes).size == 2 else None,
        "log_loss": float(log_loss(outcomes, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(outcomes, probabilities)),
        "ece": float(sum(item["games"] * abs(item["predicted"] - item["observed"]) for item in bins) / len(outcomes)),
    }


def diagnostics(matches: pd.DataFrame, probabilities) -> dict:
    outcomes = matches["blue_win"].to_numpy()
    probabilities = np.asarray(probabilities)
    fpr, tpr, _ = roc_curve(outcomes, probabilities) if np.unique(outcomes).size == 2 else ([], [], [])
    monthly = []
    for month in sorted(matches["date"].dt.strftime("%Y-%m").unique()):
        mask = matches["date"].dt.strftime("%Y-%m").eq(month).to_numpy()
        monthly.append({"month": month, **metrics(outcomes[mask], probabilities[mask])})
    return {
        "calibration": probability_bins(outcomes, probabilities),
        "roc": [{"fpr": float(x), "tpr": float(y)} for x, y in zip(fpr, tpr)],
        "monthly": monthly,
    }


def date_window(part: pd.DataFrame) -> dict:
    return {"games": len(part), "start": part["date"].min().isoformat(), "end": part["date"].max().isoformat()}


def evaluate(matches: pd.DataFrame) -> tuple[dict, pd.DataFrame, dict]:
    parts = chronological_parts(matches)
    candidates, selected, selection = {}, {}, {}
    for phase, draft in (("pre_draft", False), ("post_draft", True)):
        eligible = {name: part[part["draft_complete"]] if draft else part for name, part in parts.items()}
        candidates[phase] = fit_candidates(eligible["train"], eligible["calibration"], draft)
        validation_rows = feature_rows(eligible["validation"], draft)
        scores = {
            name: metrics(eligible["validation"]["blue_win"], model.predict_proba(validation_rows)[:, 1])
            for name, model in candidates[phase].items()
        }
        # Selection is completed using validation only, before any test predictions are made.
        selected[phase] = min(scores, key=lambda name: (scores[name]["log_loss"], name))
        selection[phase] = {"selected": selected[phase], "validation_scores": scores}

    report = {
        "schema_version": 2,
        "method": "Chronological UTC-day split: 60% base training, 10% disjoint sigmoid calibration, 15% validation/model selection, 15% test. Boundary labels unavailable at the next window's start are purged. Only finished games update history; weights stay frozen through validation/test.",
        "selection_rule": "Lowest validation log loss per phase among Logistic Regression / Random Forest, raw / sigmoid calibrated. Classification threshold fixed at 0.5.",
        "test_caveat": "This test period was inspected for the earlier baseline. These comparisons are retrospective, not a new independent external test.",
        "training": date_window(parts["train"]),
        "windows": {name: date_window(part) for name, part in parts.items()},
        "selection": selection,
        "partitions": {},
        "model_parameters": {kind: make_estimator(kind, False).get_params() for kind in MODEL_KINDS},
        "draft_logistic_C": 0.05,
    }
    predictions = []
    baseline_rate = float(parts["train"]["blue_win"].mean())
    for name in ("validation", "test"):
        part = parts[name]
        complete = part["draft_complete"].to_numpy(dtype=bool)
        draft_part = part[part["draft_complete"]]
        candidate_scores, candidate_probabilities = {}, {}
        for phase, draft, cohort in (("pre_draft", False, part), ("post_draft", True, draft_part)):
            rows = feature_rows(cohort, draft)
            candidate_probabilities[phase] = {
                tag: model.predict_proba(rows)[:, 1] for tag, model in candidates[phase].items()
            }
            candidate_scores[phase] = {
                tag: metrics(cohort["blue_win"], probabilities)
                for tag, probabilities in candidate_probabilities[phase].items()
            }
        pre = candidate_probabilities["pre_draft"][selected["pre_draft"]]
        post = candidate_probabilities["post_draft"][selected["post_draft"]]
        by_league = {
            league: metrics(part.loc[part["league"].eq(league), "blue_win"], pre[part["league"].eq(league).to_numpy()])
            for league in sorted(part["league"].unique())
        }
        report["partitions"][name] = {
            **date_window(part),
            "pre_draft": metrics(part["blue_win"], pre),
            "post_draft": metrics(draft_part["blue_win"], post),
            "pre_draft_on_complete": metrics(draft_part["blue_win"], pre[complete]),
            "constant_baseline": metrics(part["blue_win"], np.full(len(part), baseline_rate)),
            "candidates": candidate_scores,
            "by_league_pre_draft": by_league,
            "diagnostics": {"pre_draft": diagnostics(part, pre), "post_draft": diagnostics(draft_part, post)},
        }
        exported = part[["gameid", "date", "league", "patch", "blue_team", "red_team", "blue_win", "draft_complete"]].copy()
        exported["partition"] = name
        exported["pre_draft_blue_probability"] = pre
        exported["post_draft_blue_probability"] = np.nan
        exported.loc[complete, "post_draft_blue_probability"] = post
        # Save all candidates so comparisons and plots can be audited without refitting.
        for tag, probabilities in candidate_probabilities["pre_draft"].items():
            exported[f"pre_{tag}"] = probabilities
        for tag, probabilities in candidate_probabilities["post_draft"].items():
            exported[f"draft_{tag}"] = np.nan
            exported.loc[complete, f"draft_{tag}"] = probabilities
        predictions.append(exported)
    return report, pd.concat(predictions, ignore_index=True), selected


def refit_for_serving(matches: pd.DataFrame, tag: str, draft: bool) -> tuple[MatchModel, dict]:
    cohort = matches[matches["draft_complete"]] if draft else matches
    calibrated = tag.endswith("_calibrated")
    kind = tag.removesuffix("_calibrated")
    if calibrated:
        days = cohort["date"].dt.floor("D")
        cutoff = days.iloc[int(len(cohort) * 0.9)]
        train, calibration = cohort[days < cutoff], cohort[days >= cutoff]
        train = train[train["result_available_at"] < calibration["date"].min()]
    else:
        train, calibration = cohort, None
    vectorizer = DictVectorizer()
    x_train = vectorizer.fit_transform(feature_rows(train, draft))
    estimator = make_estimator(kind, draft)
    estimator.fit(x_train, train["blue_win"])
    if calibration is not None:
        estimator = CalibratedClassifierCV(FrozenEstimator(estimator), method="sigmoid")
        estimator.fit(vectorizer.transform(feature_rows(calibration, draft)), calibration["blue_win"])
    return MatchModel(vectorizer, estimator, tag, draft, calibrated), {
        "model": tag, "training": date_window(train),
        "calibration": date_window(calibration) if calibration is not None else None,
    }


def catalog_for(matches: pd.DataFrame, histories: HistoryStore) -> dict:
    teams = {}
    for row in matches.to_dict("records"):
        for side in SIDES:
            team_id = row[f"{side}_id"]
            team = teams.setdefault(team_id, {"id": team_id, "name": row[f"{side}_team"], "leagues": set()})
            team["name"] = row[f"{side}_team"]
            team["leagues"].add(row["league"])
    for team_id, team in teams.items():
        team["leagues"] = sorted(team["leagues"])
        team.update(histories.teams[team_id].snapshot())
    champions = sorted({
        row[f"{side}_{role}"] for row in matches[matches["draft_complete"]].to_dict("records")
        for side in SIDES for role in ROLES
    })
    return {
        "teams": sorted(teams.values(), key=lambda team: team["name"]),
        "champions": champions, "roles": list(ROLES),
        "leagues": sorted(matches["league"].unique()),
        "patches": sorted(matches["patch"].unique(), key=lambda value: tuple(int(part) for part in value.split("."))),
        "latest_patch": matches.iloc[-1]["patch"],
        "games": len(matches), "data_start": matches["date"].min().isoformat(),
        "data_end": matches["result_available_at"].max().isoformat(),
    }


def eda_summary(matches: pd.DataFrame, histories: HistoryStore) -> dict:
    return {
        "games": len(matches), "blue_win_rate": float(matches["blue_win"].mean()),
        "games_by_league": {str(k): int(v) for k, v in matches["league"].value_counts().items()},
        "games_by_patch": {str(k): int(v) for k, v in matches["patch"].value_counts().sort_index().items()},
        "games_by_month": {str(k): int(v) for k, v in matches["date"].dt.strftime("%Y-%m").value_counts().sort_index().items()},
        "teams": [
            {"id": key, **history.snapshot(), "losses": history.games - history.wins}
            for key, history in histories.teams.items()
        ],
        "notes": "Descriptive full-dataset summaries; not inputs for evaluating earlier games. Smoothed rates use priors; wins and losses are exact counts.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--processed-output", type=Path, default=ROOT / "data/processed/matches.csv")
    parser.add_argument("--source-label", help="Original filename when training from an immutable snapshot")
    args = parser.parse_args()
    source = args.source or default_source()
    print("Building timestamp-safe team, patch, and draft features...", flush=True)
    matches, histories, quality = build_dataset(source)
    print("Comparing raw/calibrated Logistic Regression and Random Forest...", flush=True)
    report, predictions, selected = evaluate(matches)
    with source.open("rb") as source_file:
        source_hash = hashlib.file_digest(source_file, "sha256").hexdigest()
    code_hash = hashlib.sha256(b"".join((ROOT / path).read_bytes() for path in (
        "ml/train.py", "ml/build_features.py", "ml/modeling.py",
    ))).hexdigest()
    report.update({
        "source": args.source_label or source.name, "source_sha256": source_hash, "code_sha256": code_hash,
        "quality": quality, "seed": SEED, "generated_at": datetime.now(timezone.utc).isoformat(),
        "versions": {"scikit_learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    })
    print("Refitting selected models for the historical sandbox...", flush=True)
    pre_model, pre_fit = refit_for_serving(matches, selected["pre_draft"], False)
    draft_model, draft_fit = refit_for_serving(matches, selected["post_draft"], True)
    report["serving_refit"] = {"pre_draft": pre_fit, "post_draft": draft_fit}
    bundle = {
        "schema_version": 2, "pre_model": pre_model, "draft_model": draft_model,
        "catalog": catalog_for(matches, histories), "histories": histories, "report": report,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output / "models.joblib")
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (args.output / "eda.json").write_text(json.dumps(eda_summary(matches, histories), indent=2, allow_nan=False), encoding="utf-8")
    predictions.to_csv(args.output / "holdout_predictions.csv", index=False)
    args.processed_output.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(args.processed_output, index=False)
    print(json.dumps({
        "quality": quality, "selected": selected,
        "test": {key: report["partitions"]["test"][key] for key in ("pre_draft", "post_draft", "constant_baseline")},
    }, indent=2))
    print(f"Saved models, EDA, evaluation, and auditable predictions to {args.output}")


if __name__ == "__main__":
    main()

