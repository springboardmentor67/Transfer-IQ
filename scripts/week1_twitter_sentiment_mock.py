from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

def _read_players_csv(path: Path) -> pd.DataFrame:
    content = path.read_text(encoding="utf-8").splitlines()
    lines = [ln for ln in content if ln.strip() and not ln.lstrip().startswith("#")]
    tmp = path.with_suffix(".tmp.csv")
    tmp.write_text("\n".join(lines), encoding="utf-8")
    df = pd.read_csv(tmp)
    tmp.unlink(missing_ok=True)
    return df

def generate_fake_tweet(player_name: str) -> str:
    templates = [
        f"{player_name} is absolutely amazing today! What a goal!",
        f"Can't believe {player_name} missed that chance. So frustrating.",
        f"{player_name} rumors about transfer to Real Madrid are heating up.",
        f"Is {player_name} the best player in the world? debatale.",
        f"Injury update on {player_name}: looks meaningless, should be back soon.",
        f"Love watching {player_name} play. Pure magic.",
        f"{player_name} needs to improve his defensive work rate.",
        f"Huge performance by {player_name} in the derby!",
    ]
    return random.choice(templates)

def main() -> None:
    parser = argparse.ArgumentParser(description="Week 1: Generate Mock Twitter sentiment.")
    parser.add_argument("--input", required=True, help="CSV with player_name")
    parser.add_argument("--raw-out", default="data/raw/twitter_sentiment_raw.csv/tweets.jsonl")
    parser.add_argument("--out", default="data/processed/twitter_sentiment.csv")
    parser.add_argument("--tweets-per-player", type=int, default=10)
    args = parser.parse_args()

    df_players = _read_players_csv(Path(args.input))
    if "player_name" not in df_players.columns:
        raise ValueError("Input CSV must include a 'player_name' column")

    raw_path = Path(args.raw_out)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    
    analyzer = SentimentIntensityAnalyzer()
    sentiment_rows: List[Dict[str, object]] = []

    with raw_path.open("w", encoding="utf-8") as f:
        for _, r in df_players.iterrows():
            player_name = str(r.get("player_name", "")).strip()
            if not player_name:
                continue

            for _ in range(args.tweets_per_player):
                text = generate_fake_tweet(player_name)
                created_at = (datetime.utcnow() - timedelta(days=random.randint(0, 30))).isoformat() + "Z"
                
                record = {
                    "scraped_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "player_name": player_name,
                    "query": player_name,
                    "tweet_id": str(random.randint(10**17, 10**18 - 1)),
                    "created_at": created_at,
                    "author_id": str(random.randint(1000000, 9999999)),
                    "lang": "en",
                    "text": text,
                    "retweet_count": random.randint(0, 1000),
                    "reply_count": random.randint(0, 500),
                    "like_count": random.randint(0, 5000),
                    "quote_count": random.randint(0, 100),
                    "impression_count": random.randint(1000, 100000),
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                vader = analyzer.polarity_scores(text)
                blob = TextBlob(text)
                
                sentiment_rows.append(
                    {
                        "scraped_at": record["scraped_at"],
                        "player_name": player_name,
                        "query": player_name,
                        "tweet_id": record["tweet_id"],
                        "created_at": record["created_at"],
                        "author_id": record["author_id"],
                        "lang": record["lang"],
                        "retweet_count": record["retweet_count"],
                        "reply_count": record["reply_count"],
                        "like_count": record["like_count"],
                        "quote_count": record["quote_count"],
                        "impression_count": record["impression_count"],
                        "neg": vader["neg"],
                        "neu": vader["neu"],
                        "pos": vader["pos"],
                        "compound": vader["compound"],
                        "textblob_polarity": blob.sentiment.polarity,
                        "textblob_subjectivity": blob.sentiment.subjectivity,
                    }
                )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sentiment_rows).to_csv(out_path, index=False)

    print(f"Wrote mock raw tweets: {raw_path}")
    print(f"Wrote mock sentiment table: {out_path} ({len(sentiment_rows):,} rows)")

if __name__ == "__main__":
    main()
