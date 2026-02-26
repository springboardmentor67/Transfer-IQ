from __future__ import annotations

import argparse
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv


@dataclass
class PlayerTM:
    player_name: str
    player_id: int
    slug: str


def _session() -> requests.Session:
    sess = requests.Session()
    sess.headers.update(
        {
            "User-Agent": os.getenv(
                "HTTP_USER_AGENT",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36",
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    return sess


def _sleep_polite(min_s: float = 1.0, max_s: float = 2.0) -> None:
    time.sleep(random.uniform(min_s, max_s))


def _text_to_eur(value_text: str) -> Optional[int]:
    if not value_text:
        return None
    t = value_text.strip().replace("\xa0", " ")
    m = re.search(r"€\s*([0-9]+(?:\.[0-9]+)?)\s*([mk])?", t, re.IGNORECASE)
    if not m:
        return None
    number = float(m.group(1))
    suffix = (m.group(2) or "").lower()
    mult = 1
    if suffix == "m":
        mult = 1_000_000
    elif suffix == "k":
        mult = 1_000
    return int(number * mult)


def search_player(sess: requests.Session, name: str) -> Optional[PlayerTM]:
    url = "https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche"
    resp = sess.get(url, params={"query": name}, timeout=60)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    table = soup.select_one("table.items")
    if not table:
        return None

    link = table.select_one('a[href*="/spieler/"]')
    if not link or not link.get("href"):
        return None

    href = link["href"]
    m = re.search(r"/([^/]+)/profil/spieler/(\d+)", href)
    if not m:
        m = re.search(r"/([^/]+)/.*?/spieler/(\d+)", href)
    if not m:
        return None

    slug = m.group(1)
    player_id = int(m.group(2))
    return PlayerTM(player_name=name, player_id=player_id, slug=slug)


def scrape_market_value(sess: requests.Session, p: PlayerTM) -> Dict[str, object]:
    url = f"https://www.transfermarkt.com/{p.slug}/profil/spieler/{p.player_id}"
    resp = sess.get(url, timeout=60)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    mv = soup.select_one("div.data-header__box--small .data-header__market-value-wrapper")
    mv_text = mv.get_text(" ", strip=True) if mv else ""
    eur_match = re.search(r"€\s*[0-9.,]+\s*[mk]?", mv_text, re.IGNORECASE)
    eur_text = eur_match.group(0) if eur_match else ""

    club = None
    club_el = soup.select_one("span.data-header__club")
    if club_el:
        club = club_el.get_text(" ", strip=True)

    return {
        "scraped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "player_name": p.player_name,
        "transfermarkt_player_id": p.player_id,
        "transfermarkt_slug": p.slug,
        "club": club,
        "market_value_raw": eur_text or mv_text,
        "market_value_eur": _text_to_eur(eur_text),
        "source_url": url,
    }


def scrape_market_value_history(sess: requests.Session, p: PlayerTM) -> List[Dict[str, object]]:
    url = f"https://www.transfermarkt.com/{p.slug}/marktwertverlauf/spieler/{p.player_id}"
    resp = sess.get(url, timeout=60)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    out: List[Dict[str, object]] = []
    for row in soup.select("table.items > tbody > tr"):
        cells = row.find_all("td")
        if len(cells) < 6:
            continue
        date_text = cells[0].get_text(" ", strip=True)
        value_text = cells[-1].get_text(" ", strip=True)
        out.append(
            {
                "player_name": p.player_name,
                "transfermarkt_player_id": p.player_id,
                "transfermarkt_slug": p.slug,
                "date": date_text,
                "market_value_raw": value_text,
                "market_value_eur": _text_to_eur(value_text),
                "source_url": url,
            }
        )
    return out


def scrape_injury_history(sess: requests.Session, p: PlayerTM) -> List[Dict[str, object]]:
    url = f"https://www.transfermarkt.com/{p.slug}/verletzungen/spieler/{p.player_id}"
    resp = sess.get(url, timeout=60)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    out: List[Dict[str, object]] = []
    table = soup.select_one("table.items")
    if not table:
        return out

    for row in table.select("tbody > tr"):
        cells = row.find_all("td")
        # Current layout (as of 2026-02) is typically 6 columns:
        # season, injury, from, until, days, games_missed
        if len(cells) < 6:
            continue
        out.append(
            {
                "player_name": p.player_name,
                "transfermarkt_player_id": p.player_id,
                "transfermarkt_slug": p.slug,
                "season": cells[0].get_text(" ", strip=True),
                "injury": cells[1].get_text(" ", strip=True),
                "from": cells[2].get_text(" ", strip=True),
                "until": cells[3].get_text(" ", strip=True),
                "days": cells[4].get_text(" ", strip=True),
                "games_missed": cells[5].get_text(" ", strip=True),
                "source_url": url,
            }
        )
    return out


def _read_players_csv(path: Path) -> pd.DataFrame:
    content = path.read_text(encoding="utf-8").splitlines()
    lines = [ln for ln in content if ln.strip() and not ln.lstrip().startswith("#")]
    tmp = path.with_suffix(".tmp.csv")
    tmp.write_text("\n".join(lines), encoding="utf-8")
    df = pd.read_csv(tmp)
    tmp.unlink(missing_ok=True)
    return df


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Week 1: scrape Transfermarkt market values + injury history.")
    parser.add_argument("--input", required=True, help="CSV with player_name and optional transfermarkt ids")
    parser.add_argument("--out-market", default="data/raw/transfermarkt_market_value.csv/market_values.csv")
    parser.add_argument("--out-market-history", default="data/raw/transfermarkt_market_value.csv/market_value_history.csv")
    parser.add_argument("--out-injuries", default="data/raw/injury_history.csv/injury_history.csv")
    parser.add_argument("--out-mapping", default="data/external/players_with_transfermarkt.csv")
    args = parser.parse_args()

    in_path = Path(args.input)
    df = _read_players_csv(in_path)
    if "player_name" not in df.columns:
        raise ValueError("Input CSV must include a 'player_name' column")

    sess = _session()

    mapping_rows: List[Dict[str, object]] = []
    market_rows: List[Dict[str, object]] = []
    market_history_rows: List[Dict[str, object]] = []
    injury_rows: List[Dict[str, object]] = []

    for _, r in df.iterrows():
        name = str(r.get("player_name", "")).strip()
        if not name:
            continue

        pid = r.get("transfermarkt_player_id")
        slug = r.get("transfermarkt_slug")

        player: Optional[PlayerTM] = None
        if pd.notna(pid) and pd.notna(slug) and str(pid).strip() and str(slug).strip():
            player = PlayerTM(player_name=name, player_id=int(pid), slug=str(slug).strip())
        else:
            player = search_player(sess, name)

        if not player:
            mapping_rows.append({"player_name": name, "transfermarkt_player_id": None, "transfermarkt_slug": None})
            continue

        mapping_rows.append(
            {
                "player_name": name,
                "transfermarkt_player_id": player.player_id,
                "transfermarkt_slug": player.slug,
            }
        )

        try:
            market_rows.append(scrape_market_value(sess, player))
        except Exception:
            pass
        _sleep_polite()

        try:
            market_history_rows.extend(scrape_market_value_history(sess, player))
        except Exception:
            pass
        _sleep_polite()

        try:
            injury_rows.extend(scrape_injury_history(sess, player))
        except Exception:
            pass
        _sleep_polite()

    Path(args.out_market).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_market_history).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_injuries).parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(mapping_rows).to_csv(args.out_mapping, index=False)
    pd.DataFrame(market_rows).to_csv(args.out_market, index=False)
    pd.DataFrame(market_history_rows).to_csv(args.out_market_history, index=False)
    pd.DataFrame(injury_rows).to_csv(args.out_injuries, index=False)

    print(f"Wrote mapping: {args.out_mapping} ({len(mapping_rows):,} rows)")
    print(f"Wrote market values: {args.out_market} ({len(market_rows):,} rows)")
    print(f"Wrote market value history: {args.out_market_history} ({len(market_history_rows):,} rows)")
    print(f"Wrote injury history: {args.out_injuries} ({len(injury_rows):,} rows)")


if __name__ == "__main__":
    main()
