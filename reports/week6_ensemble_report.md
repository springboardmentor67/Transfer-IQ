# TransferIQ — Week 6 Ensemble Model Report

**Project:** TransferIQ — Football Player Market Value Prediction
**Week:** 6 | **Date:** March 2026

---

## 1. What Was Built

Upgraded the Week 5 LSTM into a full LSTM + XGBoost ensemble with tier-based blending.

**Pipeline:**
- LSTM runs first → generates `lstm_pred` (market value prediction per player)
- `lstm_pred` added as a new column to the dataset
- XGBoost trained on enriched dataset using `lstm_pred` + 14 other features
- Tier-based blending: Elite players (≥€70M) → 80% LSTM + 20% XGBoost, others → 10% LSTM + 90% XGBoost
- Final output = blended ensemble prediction

---

## 2. Data Used

| Stream | Features |
|---|---|
| Performance | attacking_output_index, injury_burden_index, availability_rate, goals_per90, assists_per90, minutes_played, pass_accuracy_pct |
| Market Trends | lstm_pred, season_encoded, current_age, age_decay_factor |
| Social Sentiment | vader_compound_score, log_social_buzz (log-transformed) |

---

## 3. Validation Strategy

Train on seasons 3–4, test on season 5 (unseen). Season-based split — not random — to prevent future data leaking into training.

| Set | Seasons | Rows |
|---|---|---|
| Training | 3 and 4 | 2,000 |
| Validation | 5 (2023/24) | 1,000 |

---

## 4. Results

| Metric | LSTM | Ensemble | Change |
|---|---|---|---|
| RMSE | €3,484,678 | €2,904,173 | −16.7% ✅ |
| MAE | €2,705,656 | €2,298,892 | −15.0% ✅ |
| R² Score | 0.9935 | 0.9955 | +0.0020 ✅ |

All three metrics improved on the unseen validation set.

---

## 5. Per Tier Results

| Tier | Players | LSTM MAE | Ensemble MAE | Better |
|---|---|---|---|---|
| Low (<€10M) | 437 | €2,544,578 | €2,788,393 | LSTM |
| Mid (€10–30M) | 135 | €4,156,378 | €2,489,222 | ✅ Ensemble |
| High (€30–70M) | 227 | €2,906,949 | €1,635,022 | ✅ Ensemble |
| Elite (€70M+) | 201 | €1,854,166 | €1,856,564 | Approx. equal |

---

## 6. Feature Importance (Top 5)

| Feature | Importance | Meaning |
|---|---|---|
| lstm_pred | 58.1% | LSTM trend — dominant signal |
| log_social_buzz | 21.9% | Transfer media attention |
| minutes_played | 8.2% | Manager confidence proxy |
| attacking_output_index | 3.3% | Goals + assists |
| pass_accuracy_pct | 1.9% | Technical quality |

---

## 7. Overfitting Check

No overfitting detected. Error is consistent from training to validation season as shown in the dashboard chart. Regularisation used: `min_child_weight=20`, `reg_alpha=0.1`, `reg_lambda=1.0`.

---

## 8. Deliverables

| Deliverable | Status |
|---|---|
| XGBoost model | ✅ dashboard/xgb_model.pkl |
| Blend config | ✅ dashboard/blend_config.pkl |
| LSTM feature script | ✅ src/generate_lstm_features.py |
| Ensemble training script | ✅ src/train_xgboost.py |
| Enriched dataset | ✅ data/processed/lstm_enriched.csv |
| Streamlit dashboard | ✅ dashboard/app.py |

---

## 9. Milestone Compliance

| Requirement | Status |
|---|---|
| XGBoost ensemble implemented | ✅ |
| LSTM integrated with XGBoost | ✅ lstm_pred at 58.1% importance |
| All 3 data streams combined | ✅ 15 features |
| Tested on validation dataset | ✅ Season 5, 1,000 players |
| Ensemble improves over LSTM | ✅ RMSE −16.7%, MAE −15.0%, R² +0.002 |