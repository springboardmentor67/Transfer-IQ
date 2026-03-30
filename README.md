# 🚀 Transfer IQ
### AI-Powered Football Player Transfer Evaluation System

## 📌 Overview
Transfer IQ is an intelligent system designed to assist football managers, analysts, and scouts in evaluating player transfers using data-driven insights and predictive modeling.

The system combines:
* Machine Learning models (LSTM, XGBoost, Ensemble)
* Player statistics analysis
* Sentiment analysis over time
* Market value forecasting

It transforms raw football data into actionable transfer decisions.

## 🎯 Key Features

### 📊 Player Performance Analysis
* Historical stats visualization
* Multi-season comparison

### 🔮 Future Value Prediction
* Forecast player value for upcoming seasons
* Uses LSTM + ensemble models

### 🧠 AI-Based Decision Support
* Combines multiple models for better accuracy
* Reduces bias in transfer decisions

### 💬 Sentiment Analysis
* Tracks player public perception over time
* Uses NLP techniques

### 📈 Interactive Dashboard
* Built with Streamlit
* Easy selection of players & seasons
* Visual insights for managers

## 🧠 Models Used

| Model | Purpose |
| :--- | :--- |
| **LSTM** | Time-series forecasting of player value |
| **XGBoost** | Structured data prediction |
| **Ensemble Model** | Combines LSTM + XGBoost outputs |

## 🏗️ Project Structure

```text
Transfer-IQ/
│
├── data/                   # Raw & processed datasets
├── models/                 # Trained ML models
├── notebooks/              # Jupyter notebooks (experiments)
├── src/                    # Core source code
│   ├── preprocessing.py
│   ├── training.py
│   ├── evaluation.py
│   └── utils.py
│
├── app.py                  # Streamlit dashboard
├── requirements.txt        # Dependencies
└── README.md               # Project documentation

```
# ⚙️ Installation
## 1️⃣ Clone the repository
Bash
```
git clone [https://github.com/Vaibhav5012/Transfer-IQ.git](https://github.com/Vaibhav5012/Transfer-IQ.git)
```
## 2️⃣ Create virtual environment (recommended)

Bash
```
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
```
## 3️⃣ Install dependencies

Bash
```
pip install -r requirements.txt
```
# 📦 Dependencies
```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
tensorflow
keras
streamlit
plotly
nltk
transformers
torch
beautifulsoup4
requests
```


# 📈 Future Improvements
1 🔗 Real-time API integration (live player data)

2 🧠 Advanced transformer-based forecasting

3 📊 Team-level transfer optimization


# 🤝 Contributing
Contributions are welcome!

Bash
```
# Fork the repo
# Create a new branch
git checkout -b feature-name

# Commit changes
git commit -m "Added new feature"

# Push and create PR
```
# 📄 License
This project is open-source and available under the MIT License.

# 👤 Author
B Chiru Vaibhav (Vaibhav5012)
