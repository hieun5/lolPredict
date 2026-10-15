# Vercel hosting preparation

The public application is named **lolProPredictor**. The original hosting-preparation milestone configured a single Vercel project for the existing desktop site and 2025 prediction pipeline. A subsequent [preview milestone](vercel-preview.md) deployed and verified it on Vercel Hobby. Public production release remains pending.

## Build and serving paths

Vercel uses the FastAPI preset from the repository root. `.python-version` selects Python 3.14, the standard `app.py` entrypoint exports the hosted app, and `vercel.json` runs `python build.py` after Python dependencies are installed. `requirements.txt` remains the single dependency list; no second Python package manifest competes with it.

The hook installs the frontend lockfile with `npm ci`, builds the React site with the same-origin `/api` URL, validates the repository's existing 2025 CSV, and trains the same chronological model comparison used locally. It then verifies the resulting probabilities and source checksum. Only the serving model and a manifest are copied to `release/`; temporary training outputs are removed. No ignored local model files are needed on a clean checkout.

Training occurs during deployment builds, never during web requests. This avoids moving a Windows/Python 3.15 pickle into a different hosting runtime. The build produces the model in the environment that will serve it. Each new build trains again; it uses build time but requires no paid training service, database, or external prediction API.

The hosted app serves:

| Path | Purpose |
| --- | --- |
| `/` and `/assets/...` | Built desktop frontend, promoted to Vercel's CDN |
| `/api/catalog` | 2025 teams, leagues, champions, and cutoff |
| `/api/evaluation` | Actual historical evaluation results |
| `/api/predict` | Validated pre-draft and completed-draft estimates |
| `/api/health` | Model readiness, selected model names, and data cutoff |

Missing API routes return JSON 404s. Health checks load and verify the real model, so a missing or invalid release produces HTTP 503. The model checksum, source hash, Python minor version, and numerical library versions are checked before serving. A successful load is cached per process; runtime requests do not change the release or depend on local retraining pointers.

The static mount exposes only `frontend/dist/`. Raw CSVs, local training runs, caches, source tests, and frontend dependencies are excluded from the Python function via `vercel.json`. `.vercelignore` keeps local environments and generated files out of CLI uploads while retaining the original tracked CSV for the build. The generated release remains outside Git. The original local API/retraining workflow continues independently.

## Preview the production app locally

Use a separate Python 3.14 environment and Node.js 24. Run from the repository root:

```powershell
py -3.14 -m venv .cache/hosting-venv
.\.cache\hosting-venv\Scripts\python.exe -m pip install -r requirements.txt
.\.cache\hosting-venv\Scripts\python.exe build.py
.\.cache\hosting-venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8001
```

In another terminal:

```powershell
.\.cache\hosting-venv\Scripts\python.exe tests/smoke_api.py --base-url http://127.0.0.1:8001 --hosted
.\.cache\hosting-venv\Scripts\python.exe -m unittest discover -s tests -v
cd frontend
npm run lint
npm test
```

This preview serves the compiled frontend and production API on one origin; it does not use Vite's development proxy. Rebuild and restart this server after changing the release. The smoke check uses the catalog directly, so it requires neither raw data nor processed tables at request time.

To run the same checks against a protected Vercel preview, first link and inspect the intended project, then use the exact preview origin:

```powershell
.\.cache\hosting-venv\Scripts\python.exe tests/smoke_api.py --base-url https://YOUR-PREVIEW.vercel.app --hosted --vercel
```

This uses pinned Vercel CLI 64.2.0 and your existing CLI authentication. It leaves deployment protection enabled and sends prediction JSON through standard input. It does not print or save bypass credentials. Browser access and application checks are separate: HTTP smoke tests cannot establish visual layout or focus/scroll behavior in a real browser.

## Deployment checkpoint

The confirmed deployment target is `hieu-nguyen4/lolpropredictor` on Hobby, with **Root Directory** at the repository root, **Node.js 24**, and Python **3.14**. The repository's **FastAPI** preset and build command override the dashboard defaults through `vercel.json`. No application secrets or API keys are required. No paid integrations are configured. Hobby includes usage limits; a free public site is not unlimited hosting. Git integration is not connected yet; the verified preview was uploaded through the CLI.

With CLI deployment, first inspect the intended project from this root using `vercel project inspect --non-interactive`. Establish the correct owner/project before deploying; do not rely on automatic linking. Test a preview's health, assets, predictions, and validation errors before production. A protected preview needs authenticated access, not disabled protection. Rollback means restoring a prior deployment, including its matching model.

The actual Linux/Vercel build and hosted HTTP checks passed in the [preview milestone](vercel-preview.md). Remaining release work: review the desktop in a connected browser, stage/test production, then assign the public domain and share its URL. No cold-start benchmark or load test has been performed.

## Milestone verification

The build hook completed in a fresh local Python 3.14.8 environment with Node.js 24.21. It reproduced the previous test accuracy/AUC: pre-draft 0.588997 / 0.618964 and completed draft 0.583062 / 0.622152. The serving model is 5,248,520 bytes. All 29 required Linux Python 3.14 wheels resolved successfully; their combined uncompressed contents measured about 243.2 MiB. This is a dependency size estimate, not Vercel's final function bundle measurement.

All 34 Python regression checks passed under both Python 3.14 and the existing Python 3.15 environment. Frontend build, lint, and all seven interaction tests passed. The hosting config's keys, preset, entrypoint, duration, and exclusion length were checked against Vercel's published schema constraints.

A separate local serving folder contained Python source, compiled frontend files, and the release, with no raw CSVs, processed tables, snapshots, or local training artifacts. Real HTTP checks confirmed the page and JavaScript assets, model readiness, catalog, evaluation, pre-draft and completed-draft predictions, six invalid requests, unknown API routes, and 404s for private file paths. Before the release was added, the same app correctly returned HTTP 503 for model readiness.

Browser visual inspection remains pending because Chrome access was previously denied by the computer-use approval system. No Vercel project was linked or deployed in this milestone.

## References

- [Vercel FastAPI entrypoints, build hooks, and static mounts](https://vercel.com/docs/frameworks/backend/fastapi)
- [Supported Python versions](https://vercel.com/docs/functions/runtimes/python/python-version)
- [Python dependencies and bundle exclusions](https://vercel.com/docs/functions/runtimes/python)
- [Hobby plan and usage limits](https://vercel.com/docs/plans/hobby)
