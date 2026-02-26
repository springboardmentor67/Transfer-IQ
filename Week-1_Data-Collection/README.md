# Week 1 — Data Collection

This folder mirrors the Week 1 deliverable structure for Transfer-IQ.

## What this week covers
- Collect **player performance** data (StatsBomb Open Data)
- Collect **market value** + **injury history** data (Transfermarkt)
- Collect **sentiment** data (Twitter)
- Run initial EDA and generate basic plots

## Where the real outputs are generated
This project’s existing pipelines write outputs under:
- `data/raw/` and `data/processed/`
- `reports/` and `reports/figures/`

This Week-1 folder is a *packaging wrapper* so your GitHub repo matches the required milestone structure without committing large raw datasets.

## Run (from repo root)
```powershell
python scripts\week1_download_statsbomb.py --competition-id 43 --season-id 106 --max-matches 50
python scripts\week1_build_player_performance.py
python scripts\week1_transfermarkt_scrape.py --input data\external\players.csv
python scripts\week1_twitter_sentiment.py --input data\external\players.csv
python scripts\week1_eda.py
```

## Notes
- Large CSV/plots are intentionally ignored by git (see root `.gitignore`).
- Use the notebooks in `notebooks/` as a guided runbook.
