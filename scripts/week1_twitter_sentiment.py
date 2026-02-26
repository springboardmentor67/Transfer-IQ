from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def _read_players_csv(path: Path) -> pd.DataFrame:
    content = path.read_text(encoding="utf-8").splitlines()
    lines = [ln for ln in content if ln.strip() and not ln.lstrip().startswith("#")]
    tmp = path.with_suffix(".tmp.csv")
    tmp.write_text("\n".join(lines), encoding="utf-8")
    df = pd.read_csv(tmp)
    tmp.unlink(missing_ok=True)
    return df


def _textblob_scores(text: str) -> Dict[str, float]:
    try:
        from textblob import TextBlob

        b = TextBlob(text)
        return {
            "textblob_polarity": float(b.sentiment.polarity),
            "textblob_subjectivity": float(b.sentiment.subjectivity),
        }
    except Exception:
        return {"textblob_polarity": float("nan"), "textblob_subjectivity": float("nan")}


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Week 1: Fetch tweets via Twitter API (v2) and run VADER/TextBlob sentiment.")
    parser.add_argument("--input", required=True, help="CSV with player_name and optional twitter_query")
    parser.add_argument("--raw-out", default="data/raw/twitter_sentiment_raw.csv/tweets.jsonl")
    parser.add_argument("--out", default="data/processed/twitter_sentiment.csv")
    parser.add_argument("--max-tweets", type=int, default=None)
    args = parser.parse_args()

    bearer = os.getenv("TWITTER_BEARER_TOKEN")
    if not bearer:
        raise RuntimeError("Missing TWITTER_BEARER_TOKEN. Create a .env file (see .env.example).")

    try:
        import tweepy
    except Exception as exc:
        raise RuntimeError("tweepy is required. Install requirements.txt") from exc

    client = tweepy.Client(bearer_token=bearer, wait_on_rate_limit=True)

    df_players = _read_players_csv(Path(args.input))
    if "player_name" not in df_players.columns:
        raise ValueError("Input CSV must include a 'player_name' column")

    max_tweets_env = os.getenv("TWITTER_MAX_TWEETS_PER_QUERY")
    max_per_query = args.max_tweets or (int(max_tweets_env) if max_tweets_env else 100)

    raw_path = Path(args.raw_out)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    analyzer = SentimentIntensityAnalyzer()

    sentiment_rows: List[Dict[str, object]] = []

    with raw_path.open("w", encoding="utf-8") as f:
        for _, r in df_players.iterrows():
            player_name = str(r.get("player_name", "")).strip()
            if not player_name:
                continue
            query = str(r.get("twitter_query", "")).strip() or player_name
            q = f"({query}) lang:en -is:retweet"

            tweets = []
            paginator = tweepy.Paginator(
                client.search_recent_tweets,
                query=q,
                tweet_fields=["id", "text", "created_at", "author_id", "lang", "public_metrics"],
                max_results=100,
            )

            for resp in paginator:
                if resp.data:
                    tweets.extend(resp.data)
                if len(tweets) >= max_per_query:
                    break

            tweets = tweets[:max_per_query]

            for t in tweets:
                record = {
                    "scraped_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "player_name": player_name,
                    "query": query,
                    "tweet_id": getattr(t, "id", None),
                    "created_at": getattr(t, "created_at", None).isoformat() if getattr(t, "created_at", None) else None,
                    "author_id": getattr(t, "author_id", None),
                    "lang": getattr(t, "lang", None),
                    "text": getattr(t, "text", ""),
                }

                metrics = getattr(t, "public_metrics", None) or {}
                for k in ["retweet_count", "reply_count", "like_count", "quote_count", "impression_count"]:
                    if k in metrics:
                        record[k] = metrics[k]

                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                vader = analyzer.polarity_scores(record["text"] or "")
                tb = _textblob_scores(record["text"] or "")

                sentiment_rows.append(
                    {
                        **{k: record.get(k) for k in ["scraped_at", "player_name", "query", "tweet_id", "created_at", "author_id", "lang"]},
                        **{k: record.get(k) for k in ["retweet_count", "reply_count", "like_count", "quote_count", "impression_count"]},
                        **vader,
                        **tb,
                    }
                )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sentiment_rows).to_csv(out_path, index=False)

    print(f"Wrote raw tweets: {raw_path}")
    print(f"Wrote sentiment table: {out_path} ({len(sentiment_rows):,} rows)")


if __name__ == "__main__":
    main()
