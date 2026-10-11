from fastapi import FastAPI, HTTPException

app = FastAPI(title="lolPredictor")


@app.get("/")
def home():
    return {"message": "Welcome to lolPredictor!",
            "status": "running"
    }

@app.get("/champions/{champion_name}")
def get_champion(champion_name: str):

    champions = {
        "ahri": "mage",
        "garen": "fighter",
        "jinx": "marksman "
    }

    role = champions.get(champion_name.lower())

    if role is None:
        raise HTTPException(status_code=404,
                            detail="Champion not found")
    
    return {"champion_name": champion_name, "role": role}

