# Week 1 scripts

- `week1_download_statsbomb.py`: downloads StatsBomb Open Data (raw JSON)
- `week1_build_player_performance.py`: builds `data/processed/statsbomb_player_performance.csv`
- `week1_transfermarkt_scrape.py`: scrapes Transfermarkt market values + injury history
- `week1_twitter_sentiment.py`: fetches tweets via Twitter API and runs VADER/TextBlob
- `week1_eda.py`: creates `reports/week1_exploration_report.md` and figures

Quick run:

```powershell
python scripts\week1_download_statsbomb.py --competition-id 43 --season-id 106 --max-matches 5
python scripts\week1_build_player_performance.py
python scripts\week1_transfermarkt_scrape.py --input data\external\players.csv
python scripts\week1_eda.py
```

Twitter step needs `.env` with `TWITTER_BEARER_TOKEN`.

You can also pass the token directly:

```powershell
python scripts\week1_twitter_sentiment.py --input data\external\players.csv --bearer-token "YOUR_TOKEN"
```
