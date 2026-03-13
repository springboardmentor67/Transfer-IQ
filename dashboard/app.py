import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

# ------------------------------------------------
# Page Config
# ------------------------------------------------
st.set_page_config(
    page_title="TransferIQ Player Valuation Dashboard",
    layout="wide"
)

st.title("⚽ TransferIQ Player Valuation Dashboard")

# ------------------------------------------------
# Load Dataset
# ------------------------------------------------
df = pd.read_csv("./data/processed/player_transfer_value_with_sentiment.csv")

# ------------------------------------------------
# Load Model
# ------------------------------------------------
model = load_model("./dashboard/lstm_model.h5", compile=False)

# ------------------------------------------------
# Season Mapping
# ------------------------------------------------
season_map = {
    1: "2019/20",
    2: "2020/21",
    3: "2021/22",
    4: "2022/23",
    5: "2023/24"
}

df["season_label"] = df["season_encoded"].map(season_map)

season_order = ["2019/20","2020/21","2021/22","2022/23","2023/24"]

# ------------------------------------------------
# Sidebar
# ------------------------------------------------
st.sidebar.header("Player Selection")

player = st.sidebar.selectbox(
    "Select Player",
    sorted(df["player_name"].unique())
)

# future season selector (mentor requirement)
future_season_choice = st.sidebar.selectbox(
    "Select Future Season",
    ["2024/25", "2025/26", "2026/27"]
)

player_df = df[df["player_name"] == player].sort_values("season_encoded")

latest_season = player_df["season_label"].iloc[-1]

# ------------------------------------------------
# Metrics
# ------------------------------------------------
m_col1, m_col2, m_col3 = st.columns(3)

with m_col1:
    st.metric("Training Loss (MSE)", "0.00006")

with m_col2:
    st.metric("Model RMSE", "€3.07M")

with m_col3:
    st.metric("Model MAE", "€1.53M")

# ------------------------------------------------
# Layout
# ------------------------------------------------
col1, col2 = st.columns(2)

# ------------------------------------------------
# GRAPH 1 — Market Value Trend
# ------------------------------------------------
with col1:

    fig1 = px.line(
        player_df,
        x="season_label",
        y="market_value_eur",
        markers=True,
        title="Player Market Value Trend",
        color_discrete_sequence=["green"]
    )

    st.plotly_chart(fig1, width="stretch")

# ------------------------------------------------
# GRAPH 2 — Sentiment Trend
# ------------------------------------------------
with col2:

    fig2 = px.line(
        player_df,
        x="season_label",
        y="vader_compound_score",
        markers=True,
        title="Public Sentiment Trend",
        color_discrete_sequence=["green"]
    )

    fig2.update_layout(
        yaxis_title="Sentiment Score (-1 to 1)"
    )

    st.plotly_chart(fig2, width="stretch")

# ------------------------------------------------
# GRAPH 3 — Performance Trend
# ------------------------------------------------
with col1:

    fig3 = px.bar(
        player_df,
        x="season_label",
        y="attacking_output_index",
        title="Player Performance Trend (All Seasons)",
        color_discrete_sequence=["green"],
        text_auto='.2f'
    )

    fig3.update_layout(
        xaxis_title="Season",
        yaxis_title="Attacking Output Index",
        xaxis={'categoryorder':'array', 'categoryarray':season_order}
    )

    st.plotly_chart(fig3, width="stretch")

# ------------------------------------------------
# GRAPH 4 — LSTM Forecast
# ------------------------------------------------
with col2:

    features = [
        "market_value_eur",
        "attacking_output_index",
        "injury_burden_index",
        "availability_rate",
        "vader_compound_score",
        "social_buzz_score"
    ]

    scaler = MinMaxScaler()
    scaler.fit(df[features])

    player_scaled = scaler.transform(player_df[features])

    seq_length = 3

    season_index = player_df[player_df["season_label"] == latest_season].index[0]
    player_index = player_df.index.get_loc(season_index)

    if player_index >= seq_length - 1:

        seq = player_scaled[player_index-(seq_length-1):player_index+1]
        seq = seq.reshape(1, seq_length, len(features))

        future_predictions = []
        future_steps = 3

        for _ in range(future_steps):

            pred = model.predict(seq, verbose=0)

            pred_rescaled = scaler.inverse_transform(
                np.concatenate((pred, np.zeros((pred.shape[0], len(features)-1))), axis=1)
            )[0,0]

            # prevent negative value
            pred_rescaled = max(pred_rescaled, 0)

            # limit unrealistic season-to-season changes
            if len(future_predictions) == 0:
                last_value = player_df['market_value_eur'].iloc[-1]
            else:
                last_value = future_predictions[-1]

            max_growth = last_value * 1.40   # +40%
            min_growth = last_value * 0.70   # -30%

            pred_rescaled = min(max(pred_rescaled, min_growth), max_growth)

            future_predictions.append(pred_rescaled)

            new_row = seq[0,-1].copy()
            new_row[0] = pred[0][0]

            seq = np.concatenate([seq[:,1:,:], new_row.reshape(1,1,-1)], axis=1)

        future_seasons = ["2024/25","2025/26","2026/27"]

        connect_season = [latest_season] + future_seasons
        connect_values = [player_df['market_value_eur'].iloc[-1]] + future_predictions

        fig4 = go.Figure()

        fig4.add_trace(go.Scatter(
            x=player_df["season_label"],
            y=player_df["market_value_eur"],
            mode="lines+markers",
            name="Actual",
            line=dict(color="green")
        ))

        fig4.add_trace(go.Scatter(
            x=connect_season,
            y=connect_values,
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#b41f1f", dash="dash")
        ))

        fig4.update_layout(
            title="LSTM Multi-Step Market Value Forecast",
            xaxis_title="Season",
            yaxis_title="Value (EUR)",
            hovermode="x unified"
        )

        st.plotly_chart(fig4, width="stretch")

        # ------------------------------------------------
        # Forecast Value in Sidebar
        # ------------------------------------------------

        forecast_df = pd.DataFrame({
            "Season": future_seasons,
            "Predicted Market Value": future_predictions
        })

        selected_value = forecast_df[
            forecast_df["Season"] == future_season_choice
        ]["Predicted Market Value"].values[0]

        st.sidebar.metric(
            label=f"Predicted Value ({future_season_choice})",
            value=f"€{selected_value:,.0f}"
        )

    else:
        st.info("Not enough historical data for forecasting.")