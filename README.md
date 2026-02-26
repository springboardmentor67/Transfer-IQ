# INFOSYS-AI — Milestone 1 (Week 1)

This repo contains Week 1 deliverables: raw datasets, initial exploration (missing data + distributions), and an exploration report.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

(Optional) Copy `.env.example` to `.env` and fill in `TWITTER_BEARER_TOKEN`.

## Week 1 — Runbook

### 1) StatsBomb Open Data (player performance)

Download a competition/season from StatsBomb Open Data and store raw JSON under `data/raw/statsbomb_open_data/`.

```powershell
python scripts\week1_download_statsbomb.py --competition-id 43 --season-id 106 --max-matches 50
python scripts\week1_build_player_performance.py
```

Outputs:
- Raw: `data/raw/statsbomb_open_data/...`
- Processed: `data/processed/statsbomb_player_performance.csv`

### 2) Transfermarkt (market value + injury history)

Create or edit `data/external/players.csv` (player names and/or transfermarkt ids). Then run:

```powershell
python scripts\week1_transfermarkt_scrape.py --input data\external\players.csv
```

Outputs:
- Raw market values: `data/raw/transfermarkt_market_value.csv/market_values.csv`
- Raw injury history: `data/raw/injury_history.csv/injury_history.csv`

### 3) Twitter sentiment (VADER/TextBlob)

```powershell
python scripts\week1_twitter_sentiment.py --input data\external\players.csv
```

If you don’t want to use a `.env` file:

```powershell
python scripts\week1_twitter_sentiment.py --input data\external\players.csv --bearer-token "YOUR_TOKEN"
```

Outputs:
- Raw tweets: `data/raw/twitter_sentiment_raw.csv/tweets.jsonl`
- Sentiment table: `data/processed/twitter_sentiment.csv`

### 4) Initial EDA + exploration report

```powershell
python scripts\week1_eda.py
```

Outputs:
- Report: `reports/week1_exploration_report.md`
- Figures: `reports/figures/*.png`
