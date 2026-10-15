"""Exercise real HTTP requests through Vite's API proxy. Start both servers first."""
import csv
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:5173"


def request(path, payload=None):
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    body = json.dumps(payload).encode() if payload is not None else None
    try:
        with urlopen(Request(BASE + path, data=body, headers=headers), timeout=30) as response:
            return response.status, json.loads(response.read())
    except HTTPError as error:
        return error.code, json.loads(error.read())


with urlopen(BASE, timeout=10) as response:
    assert response.status == 200
    assert b"RiftPredict" in response.read()
status, catalog = request("/api/catalog")
assert status == 200 and catalog["games"] > 0
status, evaluation = request("/api/evaluation")
assert status == 200 and evaluation["partitions"]["test"]["pre_draft"]["games"] > 0
assert evaluation["schema_version"] == 2
for phase in ("pre_draft", "post_draft"):
    test = evaluation["partitions"]["test"]
    assert len(test["candidates"][phase]) == 4
    assert 0 <= test[phase]["roc_auc"] <= 1
    assert test["diagnostics"][phase]["roc"]
    assert test["diagnostics"][phase]["calibration"]
    assert sum(month["games"] for month in test["diagnostics"][phase]["monthly"]) == test[phase]["games"]
    choice = evaluation["selection"][phase]
    assert choice["selected"] == min(choice["validation_scores"], key=lambda tag: (choice["validation_scores"][tag]["log_loss"], tag))
with (ROOT / "data/processed/matches.csv").open(encoding="utf-8-sig", newline="") as source:
    row = next(row for row in csv.DictReader(source) if row["league"] == "LCK" and row["draft_complete"] == "True")
payload = {"league": row["league"], "blue_team_id": row["blue_id"], "red_team_id": row["red_id"]}
status, pre = request("/api/predict", payload)
assert status == 200 and pre["post_draft"] is None
assert pre["patch"] == catalog["latest_patch"]
assert pre["models"]["pre_draft"] == evaluation["selection"]["pre_draft"]["selected"]
assert 0 < pre["pre_draft"]["blue"] < 1
assert abs(sum(pre["pre_draft"].values()) - 1) < 1e-10
drafts = {f"{side}_draft": {role: row[f"{side}_{role}"] for role in ("top", "jng", "mid", "bot", "sup")} for side in ("blue", "red")}
status, post = request("/api/predict", {**payload, **drafts})
assert status == 200 and post["post_draft"] is not None
assert abs(sum(post["post_draft"].values()) - 1) < 1e-10
assert post["pre_draft"] == pre["pre_draft"]
invalid = [
    {**payload, "red_team_id": payload["blue_team_id"]},
    {**payload, "blue_draft": drafts["blue_draft"]},
    {**payload, "blue_draft": drafts["blue_draft"], "red_draft": drafts["blue_draft"]},
    {**payload, "league": "invalid"},
    {**payload, "red_team_id": "invalid"},
    {**payload, **drafts, "blue_draft": {**drafts["blue_draft"], "top": "NotAChampion"}},
]
for case in invalid:
    status, error = request("/api/predict", case)
    assert status == 422, (status, error)
print(json.dumps({
    "http_checks": "PASS: page, catalog, evaluation, pre-draft, completed draft, six invalid requests",
    "matchup": [pre["blue_team"], pre["red_team"]],
    "pre_draft": pre["pre_draft"],
    "post_draft": post["post_draft"],
    "data_end": catalog["data_end"],
}, indent=2))

