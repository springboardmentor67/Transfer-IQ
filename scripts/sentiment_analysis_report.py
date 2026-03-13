import pandas as pd

print("Running sentiment impact analysis...")

df = pd.read_csv("outputs/player_master_final.csv")

correlations = df[[
    "market_value_eur",
    "hype_score",
    "sentiment_compound",
    "injury_risk_score",
    "impact_score"
]].corr()

correlations.to_csv("outputs/sentiment_correlation_matrix.csv")

print("Correlation matrix saved.")

# Key correlations
corr_value_sentiment = correlations.loc["market_value_eur", "hype_score"]
corr_value_injury = correlations.loc["market_value_eur", "injury_risk_score"]

summary = pd.DataFrame({
    "metric": ["value_vs_hype", "value_vs_injury"],
    "correlation": [corr_value_sentiment, corr_value_injury]
})

summary.to_csv("outputs/sentiment_summary.csv", index=False)

print("Sentiment report created.")