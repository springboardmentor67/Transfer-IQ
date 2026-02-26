from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import requests
from tqdm import tqdm
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"


def _session(user_agent: Optional[str] = None) -> requests.Session:
    sess = requests.Session()
    sess.headers.update(
        {
            "User-Agent": user_agent
            or os.getenv(
                "HTTP_USER_AGENT",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36",
            ),
            "Accept": "application/json,text/html,*/*",
        }
    )
    sess.verify = False
    return sess


def _get_json(sess: requests.Session, url: str, retries: int = 5, backoff: float = 1.5) -> Any:
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            resp = sess.get(url, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(backoff**attempt)
    raise RuntimeError(f"Failed to fetch JSON after {retries} attempts: {url}") from last_exc


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def download_statsbomb_open_data(
    out_dir: Path,
    competition_id: int,
    season_id: int,
    max_matches: Optional[int] = None,
    include_360: bool = False,
    sleep_s: float = 0.3,
) -> Dict[str, Any]:
    sess = _session()

    competitions_url = f"{BASE}/competitions.json"
    competitions = _get_json(sess, competitions_url)
    _write_json(out_dir / "competitions.json", competitions)

    matches_url = f"{BASE}/matches/{competition_id}/{season_id}.json"
    matches = _get_json(sess, matches_url)
    _write_json(out_dir / "matches" / str(competition_id) / f"{season_id}.json", matches)

    match_ids = [m.get("match_id") for m in matches if m.get("match_id") is not None]
    if max_matches is not None:
        match_ids = match_ids[: max_matches]

    downloaded = {"events": 0, "lineups": 0, "three-sixty": 0}

    for match_id in tqdm(match_ids, desc="Downloading matches", unit="match"):
        events_url = f"{BASE}/events/{match_id}.json"
        lineups_url = f"{BASE}/lineups/{match_id}.json"
        events = _get_json(sess, events_url)
        lineups = _get_json(sess, lineups_url)

        _write_json(out_dir / "events" / f"{match_id}.json", events)
        _write_json(out_dir / "lineups" / f"{match_id}.json", lineups)
        downloaded["events"] += 1
        downloaded["lineups"] += 1

        if include_360:
            # Some matches don't have 360 data; treat missing as non-fatal.
            url_360 = f"{BASE}/three-sixty/{match_id}.json"
            try:
                data_360 = _get_json(sess, url_360, retries=2)
                _write_json(out_dir / "three-sixty" / f"{match_id}.json", data_360)
                downloaded["three-sixty"] += 1
            except Exception:
                pass

        time.sleep(sleep_s)

    return {
        "competition_id": competition_id,
        "season_id": season_id,
        "matches_total": len(matches),
        "matches_downloaded": len(match_ids),
        **downloaded,
        "out_dir": str(out_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 1: download StatsBomb Open Data (raw JSON).")
    parser.add_argument("--out", default="data/raw/statsbomb_open_data", help="Output folder")
    parser.add_argument("--competition-id", type=int, required=True)
    parser.add_argument("--season-id", type=int, required=True)
    parser.add_argument("--max-matches", type=int, default=None)
    parser.add_argument("--include-360", action="store_true")
    args = parser.parse_args()

    summary = download_statsbomb_open_data(
        out_dir=Path(args.out),
        competition_id=args.competition_id,
        season_id=args.season_id,
        max_matches=args.max_matches,
        include_360=args.include_360,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
