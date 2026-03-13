# 🏟️ TransferIQ: Dynamic Player Transfer Value Prediction using AI and Multi-source Data

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" />
  <img src="https://img.shields.io/badge/ML-Scikit--learn%20%7C%20XGBoost%20%7C%20LSTM-orange" />
  <img src="https://img.shields.io/badge/Internship-Infosys%20Springboard-purple" />
  <img src="https://img.shields.io/badge/Status-Active-brightgreen" />
</p>

---

## 📌 Project Overview

**TransferIQ** is an end-to-end machine learning system that dynamically predicts football player transfer market values by fusing three distinct data sources:

- **On-field performance statistics** (StatsBomb open data)
- **Injury history** (Transfermarkt)
- **Social media sentiment** (Twitter / NLP-based sentiment scoring)

The system combines classical machine learning (XGBoost, Random Forest) with a deep learning LSTM model to capture both cross-sectional patterns and temporal trends in player valuations.

---

## ❗ Problem Statement

Football transfer fees have reached unprecedented levels, yet market valuations remain opaque and highly speculative. Clubs, agents, and analysts rely on subjective assessments that ignore real-time signals like social media volume, injury recovery, or recent form.

**TransferIQ** addresses this gap by building a data-driven, interpretable model that predicts player transfer value from multi-source features — enabling smarter decisions for scouting, negotiation, and squad planning.

---

## 🗄️ Dataset Sources

| Source | Description | Format |
|---|---|---|
| [StatsBomb Open Data](https://github.com/statsbomb/open-data) | Match-level player performance stats (goals, assists, passes, pressures, etc.) | JSON → CSV |
| [Transfermarkt](https://www.transfermarkt.com) | Historical market values, contract info, injury records | Web-scraped CSV |
| Twitter / NLP Mock Data | Player mentions, tweet sentiment scores (VADER / TextBlob) | CSV |

---

## 🏗️ Project Architecture

```
TransferIQ
│
├── Data Collection          Week 1 — StatsBomb, Transfermarkt, Twitter scraping
│
├── Data Preprocessing       Week 2 — Cleaning, encoding, feature scaling
│
├── Feature Engineering      Week 3 — Sentiment pipeline, advanced metrics
│
├── Time-Series Preparation  Week 4 — Sequence generation for LSTM
│
├── Model Training           Week 5 — LSTM (PyTorch)
│                            Week 6 — Ensemble (XGBoost + Random Forest + Ridge)
│
├── Evaluation & Reporting   Week 7 — RMSE, MAE, R², visualisations
│
└── Deployment               Streamlit web app + React frontend
```

---

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data Manipulation | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Deep Learning | PyTorch (LSTM) |
| NLP / Sentiment | VADER, TextBlob |
| Visualisation | Matplotlib, Seaborn |
| Web App | Streamlit, React (Vite) |
| Version Control | Git, GitHub |
| Data Sources | StatsBomb API, Transfermarkt (scraping), Twitter API |

---

## 🤖 Model Development

### Feature Set

- **Performance features:** goals, assists, pass accuracy, progressive carries, pressures applied, match ratings
- **Injury features:** days injured per season, number of injury incidents, injury severity score
- **Sentiment features:** 30-day rolling average sentiment score, tweet volume, positive/negative ratio
- **Temporal features:** age, seasons active, career phase (rising / peak / declining)

### Models Trained

| Model | Type | Notes |
|---|---|---|
| `Ridge Regression` | Baseline linear model | Regularised least squares |
| `Random Forest Regressor` | Ensemble (bagging) | Non-linear, handles missing values |
| `XGBoost Regressor` | Ensemble (boosting) | Best performer on tabular data |
| `LSTM` | Deep learning (PyTorch) | Captures time-series value trends |
| `Stacked Ensemble` | Meta-model | Combines all above learners |

---

## 📊 Evaluation Metrics

The following metrics are used to assess regression model performance:

| Metric | Description | Interpretation |
|---|---|---|
| **RMSE** | Root Mean Squared Error | Average error in the same unit as transfer value; penalises large errors more than small ones |
| **MAE** | Mean Absolute Error | Average absolute error; easy to interpret directly in euros |
| **R²** | Coefficient of Determination | Proportion of variance explained by the model; 1.0 = perfect fit |

Evaluation code: [`scripts/model_evaluation.py`](scripts/model_evaluation.py)

---

## 📈 Visualisations

Three core visualisations are produced by [`scripts/visualizations.py`](scripts/visualizations.py):

### 1. Transfer Value Trend
A line chart showing a player's market value trajectory across seasons.

![Transfer Value Trend](reports/figures/transfer_value_trend.png)

### 2. Sentiment Score vs Market Value
A scatter plot with a regression line exploring the correlation between social media sentiment and player valuation.

![Sentiment vs Market Value](reports/figures/sentiment_vs_market_value.png)

### 3. Predicted vs Actual Values
A diagnostic scatter plot comparing model predictions against ground-truth values. Points on the dashed diagonal indicate perfect predictions.

![Predicted vs Actual](reports/figures/predicted_vs_actual.png)

---

## 🚀 How to Run the Project

### Prerequisites

- Python 3.10 or higher
- Node.js 18+ (for the React frontend only)

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/TransferIQ.git
cd TransferIQ
```

### 2. Create and Activate a Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # Windows PowerShell
# source .venv/bin/activate       # macOS / Linux
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Full Data Pipeline

```bash
python run_project.py
```

Or run individual weekly stages:

```bash
# Week 1 — Data collection
python scripts/week1_download_statsbomb.py
python scripts/week1_transfermarkt_scrape.py
python scripts/week1_twitter_sentiment.py

# Week 2 — Preprocessing
python scripts/week2_data_processing.py

# Week 3 — Sentiment pipeline
python scripts/week3_sentiment_pipeline.py

# Week 4 — Time-series prep
python scripts/week4_prep_timeseries.py
python scripts/week4_generate_sequences.py

# Week 5 — LSTM training
python scripts/week5_lstm_model.py

# Week 6 — Ensemble training
python scripts/week6_ensemble_model.py

# Week 7 — Evaluation and reporting
python scripts/week7_evaluation.py
python scripts/model_evaluation.py
python scripts/visualizations.py
```

### 5. Launch the Streamlit Dashboard

```bash
streamlit run app_streamlit.py
```

### 6. Launch the React Frontend (Optional)

```bash
cd frontend
npm install
npm run dev
```

---

## 📋 Results

> *(Update this table after final model training)*

| Model | RMSE (€) | MAE (€) | R² |
|---|---|---|---|
| Ridge Regression | — | — | — |
| Random Forest | — | — | — |
| XGBoost | — | — | — |
| LSTM | — | — | — |
| **Stacked Ensemble** | **—** | **—** | **—** |

Key findings:
- Sentiment features contributed approximately **X%** to model performance (SHAP analysis)
- Injury history reduced RMSE by **Y%** compared to performance-only baselines
- LSTM captured multi-season trends invisible to static tree-based models

---

## 🔭 Future Improvements

- [ ] Integrate live Transfermarkt and Twitter API feeds for real-time predictions
- [ ] Add SHAP explainability dashboard to the Streamlit app
- [ ] Expand to cover 500+ players across the top 5 European leagues
- [ ] Experiment with Transformer-based time-series models (Temporal Fusion Transformer)
- [ ] Add contract expiry and league competition difficulty as additional features
- [ ] Build a REST API (FastAPI) for programmatic access to predictions
- [ ] Containerise the full stack with Docker for reproducible deployment

---

## 📁 Project Structure

```
TransferIQ/
├── data/
│   ├── external/          # Static reference CSVs
│   ├── processed/         # Cleaned and feature-engineered datasets
│   └── raw/               # Original scraped / downloaded data
├── frontend/              # React + Vite frontend
├── models/                # Saved model weights (.pth, .pkl)
├── reports/
│   ├── figures/           # Generated visualisation PNGs
│   └── *.md               # Evaluation and project reports
├── scripts/               # All pipeline scripts (week1 through week7)
│   ├── model_evaluation.py   # RMSE / MAE / R² evaluation
│   └── visualizations.py     # Matplotlib + Seaborn visualisations
├── src/infosys_ai/        # Core Python package
├── app.py                 # FastAPI backend
├── app_streamlit.py       # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## 👤 Author

**Infosys Springboard Internship Project**  
Built as part of the AI/ML learning track.

---

## 📄 License

This project is for educational purposes under the Infosys Springboard internship programme.
