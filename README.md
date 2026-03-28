# 🧠 TransferIQ — AI-Powered Football Transfer Value Prediction System

> **Industry-grade, production-ready AI system** for predicting football player transfer values using LSTM + XGBoost ensemble models with explainable AI (SHAP).

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green?style=flat-square&logo=fastapi)
![React](https://img.shields.io/badge/React-19+-blue?style=flat-square&logo=react)
![Vite](https://img.shields.io/badge/Vite-7+-purple?style=flat-square&logo=vite)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=flat-square&logo=pytorch)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1+-orange?style=flat-square)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Running the Project](#-running-the-project)
- [API Documentation](#-api-documentation)
- [Model Details](#-model-details)
- [Deployment](#-deployment)

---

## 🎯 Project Overview

TransferIQ is an end-to-end machine learning system that predicts football player transfer values by analyzing:

- **Player performance data** (goals, assists, minutes, form over last 5 matches)
- **Injury history** (injury days, injury count, risk scores)
- **Sentiment analysis** (social media sentiment scores)
- **Contract data** (contract duration)

The system uses an **ensemble approach** combining XGBoost and LSTM time-series models, with SHAP-based explainability to provide transparent, interpretable predictions.

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    React + Vite Frontend                     │
│  ┌──────────┬──────────┬──────────┬──────────┬────────────┐ │
│  │Dashboard │ Compare  │Analytics │ Models   │  Alerts    │ │
│  └────┬─────┴────┬─────┴────┬─────┴────┬─────┴─────┬──────┘ │
│       │          │          │          │           │         │
└───────┼──────────┼──────────┼──────────┼───────────┼─────────┘
        │          │          │          │           │
        ▼          ▼          ▼          ▼           ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (REST)                     │
│  ┌─────────┬──────────┬──────────┬───────────┬────────────┐ │
│  │/predict │/compare  │/analytics│/model-comp│ /alerts    │ │
│  │/players │/shap     │/feature  │/recommend │ /sentiment │ │
│  └────┬────┴────┬─────┴────┬─────┴─────┬─────┴─────┬──────┘ │
│       │         │          │           │           │         │
│  ┌────▼─────────▼──────────▼───────────▼─══════════▼──────┐ │
│  │              In-Memory Cache (TTL)                      │ │
│  └────┬─────────┬──────────┬───────────┬──────────────────┘ │
└───────┼─────────┼──────────┼───────────┼────────────────────┘
        │         │          │           │
┌───────▼─────────▼──────────▼───────────▼────────────────────┐
│                    ML Model Layer                            │
│  ┌────────────┐  ┌────────────┐  ┌──────────────────────┐   │
│  │  XGBoost   │  │    LSTM    │  │  Ensemble (0.6/0.4)  │   │
│  │ (Primary)  │  │ (PyTorch)  │  │  XGBoost + RF        │   │
│  └─────┬──────┘  └─────┬──────┘  └──────────┬───────────┘   │
│        │               │                     │               │
│  ┌─────▼───────────────▼─────────────────────▼──────────┐   │
│  │              SHAP Explainer                           │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
        │
┌───────▼─────────────────────────────────────────────────────┐
│                    Data Layer                                │
│  ┌──────────────┐  ┌────────────────┐  ┌─────────────────┐  │
│  │ processed    │  │ Sequential     │  │ Model Artifacts │  │
│  │ _players.csv │  │ _sequences.csv │  │ (.pkl, .pth)    │  │
│  └──────────────┘  └────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

### 🎛 AI Dashboard
- **Player search** with dropdown autocomplete
- **Predicted transfer value** with confidence score
- **Performance trends** (area charts)
- **Player attribute radar** chart
- **SHAP feature impact** visualization

### 🧪 Explainable AI (XAI)
- **SHAP values** for every prediction
- **Feature importance** bar charts
- **"Why this value?"** natural language explanation
- Waterfall-style SHAP contribution bars

### 📊 Model Comparison
- Side-by-side metrics: RMSE, MAE, R²
- XGBoost vs Random Forest vs Linear Regression vs Ensemble
- Interactive bar charts with best-model highlighting

### 🔄 Real-Time Features
- **Server-Sent Events** (SSE) for live sentiment streaming
- Auto-updating sentiment feed
- Simulated real-time market data

### 🎯 Player Comparison
- Head-to-head comparison (2-4 players)
- Stat breakdown with visual charts
- Confidence-rated predictions per player

### 🌟 Smart Features
- **Player recommendations** based on composite scoring
- **Future value prediction** for next 1-5 seasons
- **Alert system** for value changes, injury risks, sentiment shifts

### 🚀 Production Ready
- In-memory caching with TTL
- Pydantic request/response schemas
- CORS, environment variables
- Vercel + Render deployment configs

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React 19, Vite 7, Recharts, Framer Motion, Lucide Icons |
| **Backend** | FastAPI, Uvicorn, Pydantic v2 |
| **ML Models** | XGBoost, PyTorch LSTM, Scikit-learn (LR, RF) |
| **Explainability** | SHAP (TreeExplainer) |
| **Data** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn (pipeline); Recharts (frontend) |
| **Deployment** | Vercel (frontend), Render (backend) |

---

## 📁 Project Structure

```
TransferIQ/
├── api/
│   └── main.py              # Production FastAPI server (all routes)
├── backend/
│   └── main.py              # Legacy backend (reference)
├── data/
│   ├── processed_players.csv
│   ├── processed_sequences.csv
│   └── raw/
├── frontend/
│   ├── src/
│   │   ├── App.jsx           # Main React application (6 tabs)
│   │   ├── App.css           # Premium design system
│   │   ├── index.css
│   │   └── main.jsx
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── models/
│   ├── xgboost.pkl
│   ├── linear_regression.pkl
│   ├── random_forest.pkl
│   ├── lstm_model.pth
│   ├── multivariate_lstm.pth
│   └── features.json
├── utils/
│   ├── data_pipeline.py      # Data generation & feature engineering
│   ├── models_dev.py         # Model training utilities
│   └── plotter.py            # Matplotlib visualizations
├── scripts/                  # Weekly milestone scripts
├── outputs/                  # Generated plots and reports
├── train_pipeline.py         # End-to-end training pipeline
├── requirements.txt
├── .env
├── Procfile                  # Render deployment
├── render.yaml               # Render blueprint
├── vercel.json               # Vercel frontend config
└── README.md
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10+
- Node.js 18+
- pip / npm

### 1. Clone & Install Backend

```bash
# Clone the repository
git clone https://github.com/yourusername/TransferIQ.git
cd TransferIQ

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Models

```bash
python train_pipeline.py
```

This generates:
- `data/processed_players.csv` — Cleaned player data
- `data/processed_sequences.csv` — Match sequences
- `models/*.pkl` — Trained XGBoost, RF, LR models
- `models/lstm_model.pth` — Trained LSTM model
- `models/features.json` — Feature column list
- `outputs/plots/` — Evaluation visualizations

### 3. Install Frontend

```bash
cd frontend
npm install
cd ..
```

### 4. Environment Variables

Copy `.env.example` to `.env` and configure:

```env
API_TITLE=TransferIQ Production API
API_VERSION=2.0.0
HOST=0.0.0.0
PORT=8000
CORS_ORIGINS=*
CACHE_TTL=300
VITE_API_BASE=http://localhost:8000
```

---

## ▶ Running the Project

### Start Backend (Terminal 1)
```bash
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Start Frontend (Terminal 2)
```bash
cd frontend
npm run dev
```

Open **http://localhost:5173** in your browser.

---

## 📡 API Documentation

Once the backend is running, view interactive docs at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/health` | Detailed health status |
| `POST` | `/predict` | Predict transfer value |
| `GET` | `/players` | List all players (paginated, searchable) |
| `POST` | `/compare` | Compare 2-4 players |
| `GET` | `/analytics` | Dashboard analytics data |
| `GET` | `/model-comparison` | Compare model metrics |
| `POST` | `/shap-explain` | SHAP explainability details |
| `GET` | `/feature-importance` | XGBoost feature importance |
| `POST` | `/future-prediction` | Multi-season value projection |
| `GET` | `/recommendations` | Smart player recommendations |
| `GET` | `/alerts` | Market alerts & notifications |
| `GET` | `/sentiment-stream` | SSE live sentiment feed |

### Request/Response Examples

#### POST `/predict`
```json
// Request
{
  "age": 25,
  "goals": 15,
  "assists": 8,
  "minutes": 2500,
  "injury_days": 10,
  "sentiment_score": 0.6
}

// Response
{
  "predicted_transfer_value": 45230000.0,
  "model_used": "XGBoost (Ensemble-optimized)",
  "confidence_score": 0.9234,
  "explainability": "Key drivers: form increases value (SHAP=2.34); age increases value (SHAP=1.89); goals increases value (SHAP=1.45)",
  "shap_values": { "age": 1.89, "goals": 1.45, ... },
  "feature_importance": [...]
}
```

#### POST `/compare`
```json
// Request
{ "player_ids": [1, 42] }

// Response
[
  {
    "player_id": 1,
    "name": "Player 1",
    "predicted_value": 45000000,
    "confidence": 0.92,
    ...
  },
  ...
]
```

---

## 🧠 Model Details

### XGBoost (Primary)
- 100 estimators, learning rate 0.1
- Features: age, goals, assists, minutes, form, injury_risk, sentiment_score, contract_duration, position

### LSTM (Time-Series)
- 2-layer LSTM, 64 hidden units
- Input: 5-match sequence of (goals, assists, minutes)
- PyTorch implementation

### Ensemble
- Weighted combination: 60% XGBoost + 40% Random Forest
- SHAP TreeExplainer for interpretability

### Evaluation Metrics
| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Linear Regression | ~3.2M | ~2.5M | ~0.82 |
| Random Forest | ~2.1M | ~1.6M | ~0.92 |
| XGBoost | ~1.8M | ~1.3M | ~0.95 |
| Ensemble | ~1.7M | ~1.2M | ~0.96 |

*(Values are approximate and depend on data generation seed)*

---

## 🌐 Deployment

### Frontend → Vercel
```bash
cd frontend
npm run build
# Deploy via Vercel CLI or GitHub integration
vercel --prod
```

### Backend → Render
1. Push code to GitHub
2. Create new Web Service on Render
3. Set build command: `pip install -r requirements.txt && python train_pipeline.py`
4. Set start command: `cd api && uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables from `.env`

### Using `render.yaml` Blueprint
```bash
# One-click deploy with render.yaml
# Push to GitHub, then import blueprint on Render dashboard
```

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---


