# RiftPredict

A free local web application and machine learning pipeline for professional League of Legends single-game outcomes. It compares pre-draft predictions with estimates after both champion drafts are complete.

## Run locally (PowerShell)

Run Python commands from the repository root. The existing dataset is in `data/`.

```powershell
python -m venv backend/.venv
.\backend\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\backend\.venv\Scripts\python.exe -m ml.retrain
.\backend\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
cd frontend
npm ci
npm run dev -- --host 127.0.0.1
```

Open http://127.0.0.1:5173/. The frontend proxies `/api` requests to port 8000. Retraining activates a verified model run atomically; the API picks it up without restarting. Original artifacts remain available as the `legacy` run.

Python dependencies are pinned in `requirements.txt`; the development environment uses Python 3.15. Predictions run on the local CPU with no paid APIs or accounts.

## Pipeline

1. Validate and pair the two team records for each game. Retain games with incomplete drafts for pre-draft modeling.
2. Preserve patch strings and assign the ten player-row champions to roles.
3. Compute team Elo, previous win rate, recent form, experience, team patch history, and smoothed champion-role patch history. Only games that have finished before a match starts update these histories.
4. Train Logistic Regression and Random Forest. Fit sigmoid calibrators on a later, disjoint interval.
5. Select raw or calibrated models using validation log loss. Evaluate the fixed selections and all candidates on the later test interval.
6. Refit selected model types for the historical sandbox, verify the saved artifacts, activate a versioned run, and serve predictions through FastAPI.
7. Display predictions and expandable model comparisons, ROC/reliability curves, and monthly performance on the site.

Default partitions are approximately 60% training / 10% calibration / 15% selection / 15% test, keeping UTC days together. Labels unavailable at the next partition's first game are excluded from the preceding fitting/selection window. Match outcomes, game duration, kills, gold, and other in-game statistics are excluded from model inputs. Duration only determines when an earlier game's result becomes available.

The test interval was already inspected for the first baseline. Current comparisons are retrospective; a newer season is needed for an independent external evaluation. Current accuracy and AUC are documented in [the milestone report](docs/model-reliability.md).

## Commands and generated files

```powershell
# Build only the paired feature table
.\backend\.venv\Scripts\python.exe -m ml.build_features

# Validate the existing 2025 file without training or activating anything
.\backend\.venv\Scripts\python.exe -m ml.retrain --validate-only

# Snapshot, validate, train, verify, and activate (skips unchanged data/code)
.\backend\.venv\Scripts\python.exe -m ml.retrain

# Explicitly rebuild unchanged inputs
.\backend\.venv\Scripts\python.exe -m ml.retrain --force

# Inspect the current run; reactivate a prior verified run when needed
.\backend\.venv\Scripts\python.exe -m ml.retrain --status
.\backend\.venv\Scripts\python.exe -m ml.retrain --activate RUN_ID

# Return to the original artifacts
.\backend\.venv\Scripts\python.exe -m ml.retrain --activate legacy

# Explore the raw data, including corrected wins/losses summaries
.\backend\.venv\Scripts\python.exe ml/explore_data.py

# Run regression tests
.\backend\.venv\Scripts\python.exe -m unittest discover -s tests -v

# With both servers running, exercise requests through the frontend proxy
.\backend\.venv\Scripts\python.exe tests/smoke_api.py

cd frontend
npm run build
npm run lint
```

Generated outputs are excluded from Git:

| File | Purpose |
| --- | --- |
| `data/processed/matches.csv` | Cleaned paired matches, drafts, targets, pre-match features, and result availability times |
| `data/snapshots/<sha256>.csv` | Immutable copy of the validated 2025 CSV used by a run |
| `artifacts/current.json` | Atomic pointer to the active verified model run |
| `artifacts/runs/<run-id>/models.joblib` | Serving models, histories, catalog, and evaluation metadata |
| `artifacts/runs/<run-id>/evaluation.json` | Partitions, candidate scores, plots, source/code hashes, seed, versions |
| `artifacts/runs/<run-id>/holdout_predictions.csv` | Auditable validation/test predictions from every candidate |
| `artifacts/runs/<run-id>/eda.json` | League, patch, month, team win/loss, and blue-side trend summaries |
| `artifacts/runs/<run-id>/matches.csv` | Paired feature table for that specific run |
| `artifacts/runs/<run-id>/run.json` | Dataset validation, pipeline fingerprint, model checksum, and completion/failure status |

The site uses all leagues in the supplied 2025 dataset and a competition filter. Its sandbox uses histories through the dataset cutoff and the latest recorded patch. Choosing two teams constructs a hypothetical matchup with those final histories; it does not replay a past match or predict current rosters. Completed-draft effects are historical associations.

The workflow uses the existing 2025 dataset only and performs no downloads. A failed validation, fit, or artifact check leaves the active model selection unchanged. Rollback revalidates a completed run before activation. The lower-level `python -m ml.train` command still writes standalone artifacts, but does not change an existing active run pointer.

Completed-draft submission moves focus and scrolls to the estimate after a successful prediction, respecting reduced-motion preferences.

For an eventual hosted build, configure `VITE_API_BASE_URL` at frontend build time or provide a same-origin `/api` reverse proxy. The MVP remains focused on the supplied 2025 season. Browser/mobile polish and public hosting are future milestones.

## Method references

- [Scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [Random Forest classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [FrozenEstimator](https://scikit-learn.org/stable/modules/generated/sklearn.frozen.FrozenEstimator.html)

