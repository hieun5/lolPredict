import requests
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from backend.predictions import router as prediction_router


app = FastAPI(title="lolPredict")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

DATA_DRAGON = "https://ddragon.leagueoflegends.com"

@app.get("/")
def home():
    return {"message": "Welcome to lolPredictor!",
            "status": "running"
    }

@app.get("/champions")
def get_champion(search: str = ""):
    try:
        version_response = requests.get(
            f"{DATA_DRAGON}/api/versions.json",
            timeout=10
        )
        version_response.raise_for_status()

        versions = version_response.json()
        latest_version = versions[0]
        url = (
            f"{DATA_DRAGON}/cdn/{latest_version}"
            "/data/en_US/champion.json"
        )

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        champion_data = response.json()["data"]

        champions = []

        for champion in champion_data.values():
            champions.append({
                "id": champion["id"],
                "name": champion["name"],
                "title": champion["title"],
                "tags": champion["tags"]
            })

        if search:
            matches = []

            for champion in champions:
                if search.lower() in champion["name"].lower():
                    matches.append(champion)
            champions = matches

        return {
            "version": latest_version,
            "count": len(champions),
            "champions": champions
        }

    except (requests.RequestException, ValueError, KeyError, IndexError):
        raise HTTPException(
            status_code=502,
            detail="Failed to fetch champion data"
        )


app.include_router(prediction_router)

