"""Local inference from trusted artifacts produced by python -m ml.train."""

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, model_validator

from ml.build_features import model_features, matchup_features, patch_features
from ml.artifacts import current_run, file_hash, model_path_for_run

router = APIRouter()
MODEL_PATH = Path(__file__).resolve().parents[1] / "artifacts/models.joblib"


@lru_cache(maxsize=2)
def _load_bundle_at(path: str, modified: int, expected_hash: str | None):
    model_path = Path(path)
    if expected_hash is not None and file_hash(model_path) != expected_hash:
        raise HTTPException(503, "Active model checksum is invalid. Reactivate a verified training run.")
    bundle = joblib.load(model_path)
    if bundle.get("schema_version") != 2:
        raise HTTPException(503, "Model format is outdated. Rebuild with: python -m ml.retrain")
    return bundle


def load_bundle():
    try:
        current = current_run(MODEL_PATH.parent)
        path = model_path_for_run(current["run_id"], MODEL_PATH.parent) if current else MODEL_PATH
        if not path.exists():
            raise HTTPException(503, "Models are not built yet. Run: python -m ml.retrain")
        return _load_bundle_at(str(path), path.stat().st_mtime_ns, current["model_sha256"] if current else None)
    except (OSError, ValueError) as error:
        raise HTTPException(503, "Could not load the active model. Reactivate a verified training run.") from error


class Draft(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    top: str
    jng: str
    mid: str
    bot: str
    sup: str


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    league: str
    blue_team_id: str
    red_team_id: str
    blue_draft: Draft | None = None
    red_draft: Draft | None = None

    @model_validator(mode="after")
    def validate_matchup(self):
        if self.blue_team_id == self.red_team_id:
            raise ValueError("Choose two different teams.")
        if (self.blue_draft is None) != (self.red_draft is None):
            raise ValueError("Both drafts must be complete, or both omitted.")
        if self.blue_draft is not None:
            picks = list(self.blue_draft.model_dump().values()) + list(self.red_draft.model_dump().values())
            if len(set(picks)) != 10 or any(not pick for pick in picks):
                raise ValueError("The completed draft must contain ten different champions.")
        return self


@router.get("/catalog")
def catalog(bundle: dict = Depends(load_bundle)):
    return bundle["catalog"]


@router.get("/evaluation")
def evaluation(bundle: dict = Depends(load_bundle)):
    return bundle["report"]


@router.post("/predict")
def predict(request: PredictionRequest, bundle: dict = Depends(load_bundle)):
    catalog = bundle["catalog"]
    teams = {team["id"]: team for team in catalog["teams"]}
    if request.league not in catalog["leagues"]:
        raise HTTPException(422, "Choose a league from the dataset.")
    selected = [teams.get(request.blue_team_id), teams.get(request.red_team_id)]
    if any(team is None for team in selected):
        raise HTTPException(422, "Choose teams from the dataset.")
    blue, red = selected
    if any(request.league not in team["leagues"] for team in selected):
        raise HTTPException(422, "Both teams must have played in the selected competition.")
    patch = catalog["latest_patch"]
    record = {"league": request.league, **matchup_features(blue, red)}
    record.update(patch_features(bundle["histories"], request.blue_team_id, request.red_team_id, patch))
    warnings = []
    draft_warnings = []
    if record["team_patch_coverage"] < 1:
        warnings.append(f"Limited team history on patch {patch}; patch estimates use smoothed longer-term history.")
    for side, team in (("Blue", blue), ("Red", red)):
        if team["games"] < 10:
            warnings.append(f"{side} team has only {team['games']} recorded games.")
        age = (pd.Timestamp(catalog["data_end"]) - pd.Timestamp(team["last_played"])).days
        if age > 60:
            warnings.append(f"{team['name']}'s latest game is {age} days before the dataset cutoff.")
    pre = float(bundle["pre_model"].predict_proba([model_features(record)])[0, 1])
    post = None
    if request.blue_draft is not None:
        known = set(catalog["champions"])
        for side, draft in (("blue", request.blue_draft), ("red", request.red_draft)):
            for role, champion in draft.model_dump().items():
                if champion not in known:
                    raise HTTPException(422, f"Unknown champion: {champion}")
                record[f"{side}_{role}"] = champion
        record.update(patch_features(bundle["histories"], request.blue_team_id, request.red_team_id, patch, record))
        features = model_features(record, True)
        vectorizer = bundle["draft_model"].vectorizer
        unseen = sum(
            key not in vectorizer.vocabulary_ for key in features
            if key.startswith(("role:", "pair:"))
        )
        if unseen:
            draft_warnings.append(f"{unseen} champion-role or champion-pair effects were unseen in training.")
        if record["champion_patch_coverage"] < 1:
            draft_warnings.append(f"Some champion-role picks have no recorded games on patch {patch}; their patch rates use historical priors.")
        post = float(bundle["draft_model"].predict_proba([features])[0, 1])
    return {
        "league": request.league, "patch": patch,
        "models": {"pre_draft": bundle["pre_model"].name, "post_draft": bundle["draft_model"].name},
        "blue_team": blue["name"], "red_team": red["name"],
        "pre_draft": {"blue": pre, "red": 1 - pre},
        "post_draft": {"blue": post, "red": 1 - post} if post is not None else None,
        "blue_history": {key: blue[key] for key in ("games", "wins", "elo", "win_rate", "form", "last_played")},
        "red_history": {key: red[key] for key in ("games", "wins", "elo", "win_rate", "form", "last_played")},
        "data_end": catalog["data_end"],
        "warnings": warnings, "draft_warnings": draft_warnings,
        "scope": "Historical 2025 sandbox. Models use end-of-dataset team history, not live rosters or upcoming schedules.",
    }


