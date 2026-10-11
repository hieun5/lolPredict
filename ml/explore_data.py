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

print("Dataset shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head())

# Keep only team-level records
team_data = df[df["position"] == "team"]

print("\nNumber of team records:", len(team_data))

print("\nSample team results:")
print(
    team_data[
        ["date", "league", "teamname", "side", "result"]
    ].head(10).to_string(index=False)
)


team_stats = team_data.groupby(
    ["league", "teamname"]
).agg(
    games=("result", "count"),
    wins=("result", "sum"),
    losses=("result", "sum")

)

team_stats["win_rate"] = (
    team_stats["wins"] / team_stats["games"]) * 100

team_stats = team_stats[team_stats["games"] >= 10]

team_stats = team_stats.sort_values(
    by="win_rate",
    ascending=False
)

print("\nTop teams by win rate:")
print(team_stats.head(15).round(2))

