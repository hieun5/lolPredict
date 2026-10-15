"""Build one pre-match record per game, including completed champion drafts."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
from itertools import combinations, count
import heapq
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ROLES = ("top", "jng", "mid", "bot", "sup")
SIDES = ("blue", "red")
NUMERIC_FEATURES = (
    "elo_diff", "win_rate_diff", "form_diff", "experience_diff",
    "team_patch_rate_diff", "team_patch_experience_diff", "team_patch_coverage",
)
DRAFT_NUMERIC_FEATURES = ("champion_patch_rate_diff", "champion_patch_experience_diff", "champion_patch_coverage")


@dataclass
class TeamHistory:
    games: int = 0
    wins: int = 0
    elo: float = 1500.0
    recent: deque = field(default_factory=lambda: deque(maxlen=5))
    last_played: str = ""

    def snapshot(self) -> dict:
        return {
            "games": self.games,
            "wins": self.wins,
            "elo": self.elo,
            "win_rate": (self.wins + 1) / (self.games + 2),
            "form": (sum(self.recent) + 1) / (len(self.recent) + 2),
            "last_played": self.last_played,
        }


@dataclass
class Counts:
    games: int = 0
    wins: int = 0

    def rate(self, prior: float, weight: int) -> float:
        return (self.wins + weight * prior) / (self.games + weight)

    def update(self, result: int) -> None:
        self.games += 1
        self.wins += result


@dataclass
class HistoryStore:
    teams: dict[str, TeamHistory] = field(default_factory=dict)
    team_patch: dict[tuple, Counts] = field(default_factory=dict)
    champion_role: dict[tuple, Counts] = field(default_factory=dict)
    champion_patch: dict[tuple, Counts] = field(default_factory=dict)


def patch_features(store: HistoryStore, blue_id: str, red_id: str, patch: str, drafts: dict | None = None) -> dict:
    """Use the same smoothed history lookup at training time and at serving time."""
    snapshots = [store.teams.get(key, TeamHistory()).snapshot() for key in (blue_id, red_id)]
    counts = [store.team_patch.get((key, patch), Counts()) for key in (blue_id, red_id)]
    rates = [count.rate(snapshot["win_rate"], 10) for count, snapshot in zip(counts, snapshots)]
    features = {
        "team_patch_rate_diff": rates[0] - rates[1],
        "team_patch_experience_diff": float((np.log1p(counts[0].games) - np.log1p(counts[1].games)) / 5),
        "team_patch_coverage": sum(count.games > 0 for count in counts) / 2,
    }
    if drafts is not None:
        rates, exposures = [], []
        for side in SIDES:
            side_rates, side_exposures = [], []
            for role in ROLES:
                champion = drafts[f"{side}_{role}"]
                prior = store.champion_role.get((role, champion), Counts()).rate(0.5, 20)
                count = store.champion_patch.get((patch, role, champion), Counts())
                side_rates.append(count.rate(prior, 12))
                side_exposures.append(count.games)
            rates.append(float(np.mean(side_rates)))
            exposures.append(side_exposures)
        features.update({
            "champion_patch_rate_diff": rates[0] - rates[1],
            "champion_patch_experience_diff": float((np.log1p(exposures[0]).sum() - np.log1p(exposures[1]).sum()) / 25),
            "champion_patch_coverage": sum(count > 0 for side in exposures for count in side) / 10,
        })
    return features


def default_source() -> Path:
    sources = list((ROOT / "data").glob("*.csv"))
    if len(sources) != 1:
        raise FileNotFoundError("Pass --source when data/ contains zero or multiple CSV files.")
    return sources[0]


def pair_matches(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Reject ambiguous games; retain valid games with missing drafts for pre-draft use."""
    records = []
    rejected = 0
    for game_id, group in data.groupby("gameid", sort=False):
        teams = group[group["position"].eq("team")]
        if len(teams) != 2 or set(teams["side"]) != {"Blue", "Red"}:
            rejected += 1
            continue
        teams = teams.set_index("side")
        blue = teams.loc["Blue"]
        dates = pd.to_datetime(teams["date"], errors="coerce", utc=True)
        results = pd.to_numeric(teams["result"], errors="coerce")
        if (
            dates.isna().any() or dates.nunique() != 1
            or teams["league"].isna().any() or teams["league"].nunique() != 1
            or teams["teamname"].isna().any()
            or teams["patch"].isna().any() or teams["patch"].nunique() != 1
            or not results.isin([0, 1]).all() or results.sum() != 1
        ):
            rejected += 1
            continue
        record = {
            "gameid": str(game_id), "date": dates.iloc[0],
            "league": str(blue["league"]), "patch": str(blue["patch"]),
            "blue_win": int(results.loc["Blue"]),
        }
        lengths = pd.to_numeric(teams.get("gamelength", pd.Series(dtype=float)), errors="coerce")
        valid_lengths = len(lengths) == 2 and lengths.notna().all() and np.isfinite(lengths).all() and lengths.gt(0).all()
        # Duration controls when an earlier result is knowable; it is never a model input.
        record["result_available_at"] = dates.iloc[0] + (
            pd.Timedelta(seconds=float(lengths.max())) if valid_lengths else pd.Timedelta(days=1)
        )
        champions = []
        complete = True
        for side in SIDES:
            team = teams.loc[side.title()]
            record[f"{side}_team"] = str(team["teamname"])
            record[f"{side}_id"] = (
                str(team["teamid"]) if pd.notna(team["teamid"])
                else f"name:{str(team['teamname']).strip().casefold()}"
            )
            players = group[group["side"].eq(side.title()) & group["position"].isin(ROLES)]
            valid = (
                len(players) == 5 and players["position"].nunique() == 5
                and players["champion"].notna().all()
                and players["champion"].astype(str).str.strip().ne("").all()
            )
            complete = complete and valid
            by_role = players.set_index("position")["champion"].to_dict() if valid else {}
            for role in ROLES:
                champion = str(by_role[role]).strip() if valid else ""
                record[f"{side}_{role}"] = champion
                champions.append(champion)
        if record["blue_id"] == record["red_id"]:
            rejected += 1
            continue
        record["draft_complete"] = complete and len(set(champions)) == 10
        records.append(record)
    if not records:
        raise ValueError("No valid paired matches were found in the source data.")
    matches = pd.DataFrame(records).sort_values(["date", "gameid"]).reset_index(drop=True)
    return matches, {
        "source_games": int(data["gameid"].nunique()),
        "valid_games": len(matches), "rejected_games": rejected,
        "complete_drafts": int(matches["draft_complete"].sum()),
    }


def matchup_features(blue: dict, red: dict) -> dict[str, float]:
    return {
        "elo_diff": (blue["elo"] - red["elo"]) / 400,
        "win_rate_diff": blue["win_rate"] - red["win_rate"],
        "form_diff": blue["form"] - red["form"],
        "experience_diff": float((np.log1p(blue["games"]) - np.log1p(red["games"])) / 5),
    }


def add_history(matches: pd.DataFrame) -> tuple[pd.DataFrame, HistoryStore]:
    """Only completed games update history; equal start timestamps share a snapshot."""
    store = HistoryStore()
    histories = store.teams
    records = []
    pending = []
    sequence = count()

    def apply_result(update):
        _, _, team_id, result, delta, side, record = update
        history = histories[team_id]
        history.games += 1
        history.wins += result
        history.elo += delta
        history.recent.append(result)
        history.last_played = max(history.last_played, record["date"].isoformat())
        store.team_patch.setdefault((team_id, record["patch"]), Counts()).update(result)
        if record["draft_complete"]:
            for role in ROLES:
                champion = record[f"{side}_{role}"]
                store.champion_role.setdefault((role, champion), Counts()).update(result)
                store.champion_patch.setdefault((record["patch"], role, champion), Counts()).update(result)

    for date, batch in matches.groupby("date", sort=True):
        while pending and pending[0][0] < date:
            apply_result(heapq.heappop(pending))
        for record in batch.to_dict("records"):
            blue = histories.setdefault(record["blue_id"], TeamHistory()).snapshot()
            red = histories.setdefault(record["red_id"], TeamHistory()).snapshot()
            record.update(matchup_features(blue, red))
            record.update(patch_features(
                store, record["blue_id"], record["red_id"], record["patch"],
                record if record["draft_complete"] else None,
            ))
            for side, snapshot in (("blue", blue), ("red", red)):
                for name, value in snapshot.items():
                    record[f"{side}_{name}_before"] = value
            expected = 1 / (1 + 10 ** ((red["elo"] - blue["elo"]) / 400))
            delta = 24 * (record["blue_win"] - expected)
            for side, result, change in (("blue", record["blue_win"], delta), ("red", 1 - record["blue_win"], -delta)):
                heapq.heappush(pending, (
                    record["result_available_at"], next(sequence), record[f"{side}_id"], result, change, side, record,
                ))
            records.append(record)
    while pending:
        apply_result(heapq.heappop(pending))
    return pd.DataFrame(records), store


def model_features(record: dict, include_draft: bool = False) -> dict:
    features = {name: float(record[name]) for name in NUMERIC_FEATURES}
    features["league"] = record["league"]
    if include_draft:
        features.update({name: float(record[name]) for name in DRAFT_NUMERIC_FEATURES})
        # Shared weights across blue/red sides. Pair effects are associations, not causal strength.
        for side, sign in (("blue", 1), ("red", -1)):
            champions = []
            for role in ROLES:
                champion = record[f"{side}_{role}"]
                champions.append(champion)
                key = f"role:{role}:{champion}"
                features[key] = features.get(key, 0) + sign
            for pair in combinations(sorted(champions), 2):
                key = f"pair:{pair[0]}|{pair[1]}"
                features[key] = features.get(key, 0) + sign
    return features


def build_dataset(source: Path) -> tuple[pd.DataFrame, HistoryStore, dict]:
    columns = [
        "gameid", "date", "league", "patch", "position", "side",
        "teamname", "teamid", "champion", "result", "gamelength",
    ]
    data = pd.read_csv(source, usecols=columns, dtype={"patch": "string"}, low_memory=False)
    matches, quality = pair_matches(data)
    matches, histories = add_history(matches)
    return matches, histories, quality


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/matches.csv")
    args = parser.parse_args()
    matches, _, quality = build_dataset(args.source or default_source())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(args.output, index=False)
    print(quality)
    print(f"Saved {len(matches):,} match records to {args.output}")


if __name__ == "__main__":
    main()
