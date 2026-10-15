"""Same-origin production app: built desktop site and read-only predictions."""

from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from backend.predictions import load_bundle, router
from backend.release import load_release

ROOT = Path(__file__).resolve().parent
app = FastAPI(title="lolProPredictor", docs_url=None, redoc_url=None, openapi_url=None)
app.dependency_overrides[load_bundle] = load_release
app.include_router(router, prefix="/api")


@app.get("/api/health")
def health(bundle: dict = Depends(load_release)):
    return {
        "status": "ready", "season": 2025,
        "data_end": bundle["catalog"]["data_end"],
        "models": {"pre_draft": bundle["pre_model"].name, "post_draft": bundle["draft_model"].name},
    }


# Register the API first so unknown /api paths also return JSON 404s.
@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
def unknown_api(path: str):
    raise HTTPException(404, "API route not found.")


# Vercel promotes this directory to its CDN; it also works under local Uvicorn.
# check_dir=False allows framework discovery before the build creates dist/.
app.mount("/", StaticFiles(directory=ROOT / "frontend/dist", html=True, check_dir=False), name="site")
