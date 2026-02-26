from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_get(d: Dict[str, Any], *keys: str, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _match_max_minute(events: List[Dict[str, Any]]) -> int:
    max_min = 0
    for ev in events:
        minute = ev.get("minute")
        if isinstance(minute, int):
            max_min = max(max_min, minute)
    # add a small buffer
    if max_min <= 90:
        return 90
    if max_min <= 120:
        return 120
    return max_min


def _minutes_from_lineups(lineups: List[Dict[str, Any]], match_minutes: int) -> Dict[Tuple[int, str], float]:
    """Return minutes played keyed by (player_id, player_name)."""
    minutes: Dict[Tuple[int, str], float] = defaultdict(float)
    for team in lineups:
        for p in team.get("lineup", []) or []:
            player_id = _safe_get(p, "player_id")
            player_name = _safe_get(p, "player_name")
            if player_id is None or player_name is None:
                continue

            positions = p.get("positions") or []
            if not positions:
                # If no positions data, assume 0 (we'll still collect events-based stats)
                continue

            for pos in positions:
                start = pos.get("from")
                end = pos.get("to")
                if start is None:
                    continue
                if end is None:
                    end = match_minutes
                try:
                    minutes[(int(player_id), str(player_name))] += max(0, float(end) - float(start))
                except Exception:
                    continue

    return minutes


def build_player_match_stats(raw_dir: Path) -> pd.DataFrame:
    matches_files = sorted((raw_dir / "matches").rglob("*.json"))
    if not matches_files:
        raise FileNotFoundError(f"No matches files found under: {raw_dir / 'matches'}")

    rows: List[Dict[str, Any]] = []

    for matches_path in matches_files:
        matches = _read_json(matches_path)
        # parse competition/season from path .../matches/{competition_id}/{season_id}.json
        try:
            season_id = int(matches_path.stem)
            competition_id = int(matches_path.parent.name)
        except Exception:
            season_id = None
            competition_id = None

        for m in matches:
            match_id = m.get("match_id")
            if match_id is None:
                continue

            events_path = raw_dir / "events" / f"{match_id}.json"
            lineups_path = raw_dir / "lineups" / f"{match_id}.json"
            if not events_path.exists() or not lineups_path.exists():
                continue

            events = _read_json(events_path)
            lineups = _read_json(lineups_path)

            match_minutes = _match_max_minute(events)
            minutes_played = _minutes_from_lineups(lineups, match_minutes)

            # event aggregations by player
            agg: Dict[Tuple[int, str], Dict[str, float]] = defaultdict(lambda: defaultdict(float))

            for ev in events:
                player = ev.get("player") or {}
                player_id = player.get("id")
                player_name = player.get("name")
                if player_id is None or player_name is None:
                    continue
                key = (int(player_id), str(player_name))

                ev_type = _safe_get(ev, "type", "name")
                if ev_type == "Pass":
                    agg[key]["passes"] += 1
                    if _safe_get(ev, "pass", "outcome") is None:
                        agg[key]["passes_completed"] += 1
                    if _safe_get(ev, "pass", "goal_assist") is True:
                        agg[key]["assists"] += 1
                    if _safe_get(ev, "pass", "shot_assist") is True:
                        agg[key]["key_passes"] += 1

                elif ev_type == "Shot":
                    agg[key]["shots"] += 1
                    if _safe_get(ev, "shot", "outcome", "name") == "Goal":
                        agg[key]["goals"] += 1
                    xg = _safe_get(ev, "shot", "statsbomb_xg")
                    if isinstance(xg, (int, float)):
                        agg[key]["xg"] += float(xg)

                elif ev_type == "Duel":
                    duel_type = _safe_get(ev, "duel", "type", "name")
                    if duel_type == "Tackle":
                        agg[key]["tackles"] += 1

                elif ev_type == "Interception":
                    agg[key]["interceptions"] += 1

                elif ev_type == "Foul Committed":
                    agg[key]["fouls_committed"] += 1

            # build rows (include players present in minutes or events)
            keys = set(agg.keys()) | set(minutes_played.keys())
            for (player_id, player_name) in keys:
                row = {
                    "match_id": match_id,
                    "competition_id": competition_id,
                    "season_id": season_id,
                    "match_date": m.get("match_date"),
                    "home_team": _safe_get(m, "home_team", "home_team_name"),
                    "away_team": _safe_get(m, "away_team", "away_team_name"),
                    "player_id": player_id,
                    "player_name": player_name,
                    "minutes": minutes_played.get((player_id, player_name), 0.0),
                }
                row.update({k: float(v) for k, v in agg.get((player_id, player_name), {}).items()})
                rows.append(row)

    df = pd.DataFrame(rows).fillna(0)
    # normalize date
    if "match_date" in df.columns:
        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    return df


def build_player_season_stats(match_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["competition_id", "season_id", "player_id", "player_name"]
    numeric_cols = [
        c
        for c in match_df.columns
        if c
        not in {
            "match_id",
            "match_date",
            "home_team",
            "away_team",
            "player_name",
        }
        and c not in set(group_cols)
        and pd.api.types.is_numeric_dtype(match_df[c])
    ]
    agg_map = {c: "sum" for c in numeric_cols}

    df = (
        match_df.groupby(group_cols, dropna=False)
        .agg(agg_map)
        .reset_index()
        .sort_values(["competition_id", "season_id", "player_name"])
    )

    # per-90 rates (avoid divide by zero)
    mins = df["minutes"].replace(0, pd.NA)
    for col in ["goals", "assists", "shots", "passes", "xg", "key_passes"]:
        if col in df.columns:
            df[f"{col}_per90"] = (df[col] / mins) * 90
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 1: build player performance dataset from StatsBomb raw JSON.")
    parser.add_argument("--raw", default="data/raw/statsbomb_open_data", help="Raw StatsBomb folder")
    parser.add_argument("--out-dir", default="data/processed", help="Output folder")
    args = parser.parse_args()

    raw_dir = Path(args.raw)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    match_df = build_player_match_stats(raw_dir)
    match_out = out_dir / "statsbomb_player_match_stats.csv"
    match_df.to_csv(match_out, index=False)

    season_df = build_player_season_stats(match_df)
    season_out = out_dir / "statsbomb_player_performance.csv"
    season_df.to_csv(season_out, index=False)

    print(f"Wrote {match_out} ({len(match_df):,} rows)")
    print(f"Wrote {season_out} ({len(season_df):,} rows)")


if __name__ == "__main__":
    main()
