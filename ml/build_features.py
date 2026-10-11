from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
data_folder = project_root / "data"

csv_files = list(data_folder.glob("*.csv"))

if len(csv_files) != 1:
    raise FileNotFoundError(
        "Expected exactly one CSV file in the data folder."
    )

df = pd.read_csv(csv_files[0], low_memory=False)

team_data = df[df["position"] == "team"].copy()

team_data["date"] = pd.to_datetime(
    team_data["date"], errors="coerce"
)

team_data["result"] = pd.to_numeric(
    team_data["result"], errors="coerce"
)

team_data = team_data.dropna(
    subset=["date", "league", "teamname", "result"])


team_data = team_data[
    team_data["result"].isin([0, 1]) 
].copy()

team_data = team_data.sort_values(
    by=["date", "gameid"]
).reset_index(drop=True)


groups = team_data.groupby(["league", "teamname"])


team_data["games_before"] = groups.cumcount()


team_data["wins_before"] = (
    groups["result"].cumsum() - team_data["result"]
)


team_data["win_rate_before"] = (
    (team_data["wins_before"] + 1)
    / (team_data["games_before"] + 2)
)


columns = [
    "date",
    "teamname",
    "games_before",
    "wins_before",
    "win_rate_before",
    "result"
]

print("\nHistorical team features:")
print(team_data[columns].head(15).to_string(index=False))
