"""Validate the existing 2025 Oracle's Elixir dataset before fitting any models."""

from pathlib import Path

import numpy as np
import pandas as pd

from ml.artifacts import file_hash
from ml.build_features import ROLES

SOURCE_COLUMNS = (
    "gameid", "date", "league", "patch", "position", "side",
    "teamname", "teamid", "champion", "result", "gamelength",
)


def validate_dataset(source: Path) -> dict:
    try:
        header = pd.read_csv(source, nrows=0)
        missing = sorted(set(SOURCE_COLUMNS) - set(header.columns))
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")
        data = pd.read_csv(source, usecols=list(SOURCE_COLUMNS), dtype={"patch": "string"}, low_memory=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as error:
        raise ValueError("The source must be a readable Oracle's Elixir CSV.") from error
    if data.empty:
        raise ValueError("The dataset contains no match rows.")
    for column in ("gameid", "date", "league", "patch", "position", "side", "teamname"):
        if data[column].isna().any() or data[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing values in required field: {column}")
    if not data["position"].isin(["team", *ROLES]).all() or not data["side"].isin(["Blue", "Red"]).all():
        raise ValueError("Unexpected position or side values.")
    if data.duplicated(["gameid", "side", "position"]).any():
        raise ValueError("Duplicate game/side/role records found.")
    dates = pd.to_datetime(data["date"], utc=True, errors="coerce")
    if dates.isna().any() or not dates.dt.year.eq(2025).all():
        raise ValueError("This workflow expects valid dates from the existing 2025 season.")
    if not data["patch"].str.fullmatch(r"15\.\d{1,2}").all():
        raise ValueError("Expected 2025 patch strings such as 15.01 or 15.10.")
    data["result"] = pd.to_numeric(data["result"], errors="coerce")
    if not data["result"].isin([0, 1]).all():
        raise ValueError("Every completed match row must have a binary result.")
    lengths = pd.to_numeric(data["gamelength"], errors="coerce")
    if lengths.isna().any() or not np.isfinite(lengths).all() or not lengths.gt(0).all():
        raise ValueError("Every match needs a positive, finite game duration.")
    data["date"] = dates
    grouped = data.groupby("gameid")
    if not grouped.size().eq(12).all():
        raise ValueError("Expected two team records and ten player records per game.")
    for column in ("date", "league", "patch", "gamelength"):
        if not grouped[column].nunique().eq(1).all():
            raise ValueError(f"Conflicting {column} values within a game.")
    sides = data.groupby(["gameid", "side"])
    if not sides["result"].nunique().eq(1).all():
        raise ValueError("Player and team results disagree within a side.")
    teams = data[data["position"].eq("team")]
    if not teams.groupby("gameid")["result"].sum().eq(1).all():
        raise ValueError("Every game must have exactly one winning side.")
    if not sides["teamname"].nunique().eq(1).all():
        raise ValueError("Conflicting team names within a side.")
    if not teams.groupby("gameid")["teamname"].nunique().eq(2).all():
        raise ValueError("A team cannot play itself.")
    players = data[data["position"].isin(ROLES)]
    picks = players.groupby("gameid")["champion"]
    complete = int((picks.count().eq(10) & picks.nunique().eq(10)).sum())
    return {
        "source_sha256": file_hash(source), "bytes": source.stat().st_size,
        "rows": len(data), "games": int(data["gameid"].nunique()),
        "complete_drafts": complete, "incomplete_drafts": int(data["gameid"].nunique()) - complete,
        "start": dates.min().isoformat(), "end": dates.max().isoformat(),
        "leagues": sorted(data["league"].unique()),
        "patches": sorted(data["patch"].unique()),
        "missing_team_ids": int(teams["teamid"].isna().sum()),
    }
