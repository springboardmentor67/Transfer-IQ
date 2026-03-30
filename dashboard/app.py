# ============================================================
# FILE: dashboard/app.py
# Week 8 — Final UI with navbar, search, player detail page
# ============================================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(
    page_title="TransferIQ",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------
# Custom CSS — navbar, footer, search card
# ------------------------------------------------
st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0rem; padding-bottom: 0rem; }

/* full page min height so footer stays at bottom */
[data-testid="stAppViewContainer"] {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
}
[data-testid="stVerticalBlock"] {
    flex: 1;
}

.navbar {
    background: #1a50a3;
    padding: 14px 40px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0;
}
.navbar-brand { color: white; font-size: 22px; font-weight: 700; letter-spacing: 1px; }
.navbar-sub   { color: rgba(255,255,255,0.7); font-size: 13px; }

.footer {
    background: #1a50a3;
    color: rgba(255,255,255,0.7);
    text-align: center;
    padding: 14px;
    font-size: 12px;
    margin-top: 40px;
    width: 100%;
}

.metric-card {
    background: #f0f4ff;
    border-left: 4px solid #1a50a3;
    border-radius: 8px;
    padding: 18px 24px;
    margin-bottom: 16px;
}
.metric-label { font-size: 13px; color: #666; margin-bottom: 4px; }
.metric-value { font-size: 36px; font-weight: 700; color: #1a50a3; }
.metric-sub   { font-size: 12px; color: #888; margin-top: 4px; }

.player-header {
    background: linear-gradient(135deg, #1a50a3 0%, #1a7340 100%);
    color: white; padding: 30px 40px;
    border-radius: 12px; margin-bottom: 24px;
}
.player-name-big { font-size: 28px; font-weight: 700; margin-bottom: 6px; }
.player-meta { font-size: 14px; opacity: 0.85; }

/* season button active state */
div[data-testid="stButton"] button[kind="secondary"].season-active {
    background: #1a50a3 !important;
    color: white !important;
    border-color: #1a50a3 !important;
}

/* search result list */
.player-result {
    padding: 10px 14px;
    border: 0.5px solid #ddd;
    border-radius: 6px;
    margin-bottom: 6px;
    cursor: pointer;
    font-size: 14px;
    background: white;
}
.player-result:hover { background: #f0f4ff; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# Navbar
# ------------------------------------------------
st.markdown("""
<div class="navbar">
    <div class="navbar-brand">⚽ TransferIQ</div>
    <div class="navbar-sub">Football Player Market Value Prediction</div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------
# Load data & models (cached)
# ------------------------------------------------
@st.cache_resource
def load_models():
    lstm = load_model("./dashboard/lstm_model.h5", compile=False)
    xgb  = joblib.load("./dashboard/xgb_model.pkl")
    return lstm, xgb

@st.cache_data
def load_data():
    df = pd.read_csv("./data/processed/player_transfer_value_with_sentiment.csv")
    df = df.sort_values(["player_name", "season_encoded"]).reset_index(drop=True)
    season_map = {1:"2019/20", 2:"2020/21", 3:"2021/22", 4:"2022/23", 5:"2023/24"}
    df["season_label"] = df["season_encoded"].map(season_map)
    return df

lstm_model, xgb_model = load_models()
df = load_data()

lstm_features = ["market_value_eur","attacking_output_index","injury_burden_index",
                 "availability_rate","vader_compound_score","social_buzz_score"]
xgb_features  = ["lstm_pred","current_age","age_decay_factor","position_encoded",
                  "season_encoded","attacking_output_index","injury_burden_index",
                  "availability_rate","goals_per90","assists_per90",
                  "goal_contributions_per90","minutes_played","pass_accuracy_pct",
                  "vader_compound_score","log_social_buzz"]

scaler = MinMaxScaler()
scaler.fit(df[lstm_features])

all_players = sorted(df["player_name"].unique())

# ------------------------------------------------
# Session state — track which page we're on
# ------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page   = "home"
if "player" not in st.session_state:
    st.session_state.player = None
if "season" not in st.session_state:
    st.session_state.season = "2024/25"

# ------------------------------------------------
# Ensemble prediction helper
# ------------------------------------------------
def get_ensemble_preds(player_name, future_seasons_all):
    p_df      = df[df["player_name"] == player_name].sort_values("season_encoded").reset_index(drop=True)
    last_val  = float(p_df["market_value_eur"].iloc[-1])
    latest_s  = p_df["season_label"].iloc[-1]

    scaled = scaler.transform(p_df[lstm_features])
    seq    = scaled[-3:].copy()
    lstm_raw_list, lstm_cap_list = [], []

    for step in range(3):
        ps  = lstm_model.predict(seq.reshape(1,3,len(lstm_features)), verbose=0)
        pad = np.zeros((1, len(lstm_features)-1))
        pe  = float(scaler.inverse_transform(np.concatenate([ps,pad],axis=1))[0,0])
        pe  = max(pe, 0)
        lstm_raw_list.append(pe)
        lv  = last_val if step==0 else lstm_cap_list[-1]
        lstm_cap_list.append(float(np.clip(pe, lv*0.70, lv*1.40)))
        nr = seq[-1].copy(); nr[0] = float(ps[0][0])
        seq = np.vstack([seq[1:], nr])

    latest_row = p_df.iloc[-1]
    ens_list   = []
    for s_idx in range(3):
        fd = {
            "lstm_pred": lstm_raw_list[s_idx],
            "current_age": latest_row.get("current_age", np.nan),
            "age_decay_factor": latest_row.get("age_decay_factor", np.nan),
            "position_encoded": latest_row.get("position_encoded", np.nan),
            "season_encoded": 5 + s_idx + 1,
            "attacking_output_index": latest_row.get("attacking_output_index", np.nan),
            "injury_burden_index": latest_row.get("injury_burden_index", np.nan),
            "availability_rate": latest_row.get("availability_rate", np.nan),
            "goals_per90": latest_row.get("goals_per90", np.nan),
            "assists_per90": latest_row.get("assists_per90", np.nan),
            "goal_contributions_per90": latest_row.get("goal_contributions_per90", np.nan),
            "minutes_played": latest_row.get("minutes_played", np.nan),
            "pass_accuracy_pct": latest_row.get("pass_accuracy_pct", np.nan),
            "vader_compound_score": latest_row.get("vader_compound_score", np.nan),
            "log_social_buzz": float(np.log1p(latest_row.get("social_buzz_score", 0))),
        }
        cols = [f for f in xgb_features if f in fd]
        xin  = pd.DataFrame([[fd[f] for f in cols]], columns=cols)
        xv   = float(max(xgb_model.predict(xin)[0], 0))
        lv   = lstm_cap_list[s_idx]
        ev   = 0.8*lv + 0.2*xv if last_val >= 70e6 else 0.1*lv + 0.9*xv
        ev   = float(np.clip(max(ev,0), last_val*0.60, last_val*1.40))
        ens_list.append(ev)

    return p_df, last_val, latest_s, ens_list

# ================================================
# HOME PAGE
# ================================================
if st.session_state.page == "home":

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 3, 1])
    with col_c:
        st.markdown("### 🔍 Search Player")

        # Native selectbox — has built-in search/filter as you type
        selected_player = st.selectbox(
            "Player",
            options=[""] + all_players,
            index=0,
            placeholder="Type to search player...",
            format_func=lambda x: "Type to search player..." if x == "" else x,
            label_visibility="collapsed",
        )
        if selected_player == "":
            selected_player = None

        st.markdown("<br>", unsafe_allow_html=True)

        # Season selector with active color
        st.markdown("**Select Future Season**")
        season_cols = st.columns(3)
        for i, s in enumerate(["2024/25", "2025/26", "2026/27"]):
            with season_cols[i]:
                is_active = (st.session_state.season == s)
                label = f"✓ {s}" if is_active else s
                btn_t = "primary" if is_active else "secondary"
                if st.button(label, key=f"season_{s}",
                             type=btn_t, use_container_width=True):
                    st.session_state.season = s
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        if selected_player:
            if st.button("🔎 View Player Analysis", type="primary", use_container_width=True):
                st.session_state.player = selected_player
                st.session_state.page   = "player"
                st.rerun()
        else:
            st.button("🔎 View Player Analysis", type="primary",
                      use_container_width=True, disabled=True)

# ================================================
# PLAYER DETAIL PAGE
# ================================================
elif st.session_state.page == "player":

    player  = st.session_state.player
    season  = st.session_state.season
    fut_all = ["2024/25", "2025/26", "2026/27"]
    fut_idx = fut_all.index(season)

    # Back button
    if st.button("← Back to Search"):
        st.session_state.page = "home"
        st.rerun()

    # Get predictions
    with st.spinner("Generating prediction..."):
        p_df, last_val, latest_s, ens_preds = get_ensemble_preds(player, fut_all)

    pred_val    = ens_preds[fut_idx]
    pct_change  = (pred_val - last_val) / last_val * 100
    pct_str     = f"+{pct_change:.1f}%" if pct_change >= 0 else f"{pct_change:.1f}%"
    position    = p_df["position"].iloc[-1] if "position" in p_df.columns else "Player"
    age         = int(p_df["current_age"].iloc[-1]) if "current_age" in p_df.columns else ""

    # ── PLAYER HEADER ──
    st.markdown(f"""
    <div class="player-header">
        <div class="player-name-big">{player}</div>
        <div class="player-meta">{position} &nbsp;|&nbsp; Age: {age} &nbsp;|&nbsp; Season: {season}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── TOP METRICS ──
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Predicted Market Value ({season})</div>
            <div class="metric-value">€{pred_val/1e6:.1f}M</div>
            <div class="metric-sub">Ensemble model prediction</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Last Known Value (2023/24)</div>
            <div class="metric-value">€{last_val/1e6:.1f}M</div>
            <div class="metric-sub">Season 5 actual value</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        color = "#1a7340" if pct_change >= 0 else "#b41f1f"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Projected Change</div>
            <div class="metric-value" style="color:{color}">{pct_str}</div>
            <div class="metric-sub">vs last known value</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── GRAPHS ──
    col1, col2 = st.columns(2)

    # GRAPH 1 — Historical Market Value
    with col1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=p_df["season_label"], y=p_df["market_value_eur"],
            mode="lines+markers", fill="tozeroy",
            line=dict(color="#1a7340", width=2), marker=dict(size=9),
            fillcolor="rgba(26,115,64,0.08)",
            hovertemplate="Season: %{x}<br>Value: €%{y:,.0f}<extra></extra>",
        ))
        for _, row in p_df.iterrows():
            fig1.add_annotation(x=row["season_label"], y=row["market_value_eur"],
                text=f"€{row['market_value_eur']/1e6:.1f}M", showarrow=False,
                yshift=14, font=dict(size=11, color="#1a7340"))
        fig1.update_layout(title="Historical Market Value", xaxis_title="Season",
            yaxis_title="Value (EUR)", showlegend=False, hovermode="x unified")
        st.plotly_chart(fig1, width='stretch')

    # GRAPH 2 — Ensemble Forecast
    with col2:
        fore_x = [latest_s] + fut_all
        fore_y = [last_val]  + ens_preds
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=fore_x, y=fore_y, mode="lines+markers",
            name="Ensemble Forecast",
            line=dict(color="#1a50a3", width=2.5),
            marker=dict(size=9, symbol="diamond"),
            hovertemplate="Season: %{x}<br>Forecast: €%{y:,.0f}<extra></extra>",
        ))
        for sx, sy in zip(fore_x[1:], fore_y[1:]):
            fig2.add_annotation(x=sx, y=sy, text=f"€{sy/1e6:.1f}M",
                showarrow=False, yshift=14, font=dict(size=11, color="#1a50a3"))
        fig2.add_vrect(x0=season, x1=season,
            fillcolor="rgba(26,80,163,0.08)", layer="below", line_width=0)
        fig2.update_layout(title="Ensemble Market Value Forecast",
            xaxis_title="Season", yaxis_title="Value (EUR)",
            showlegend=False, hovermode="x unified")
        st.plotly_chart(fig2, width='stretch')

    # GRAPH 3 — Performance Trends
    with col1:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=p_df["season_label"], y=p_df["goals_per90"],
            mode="lines+markers", name="Goals/90",
            line=dict(color="#b41f1f", width=2), marker=dict(size=7),
            hovertemplate="%{x}: %{y:.2f}<extra>Goals/90</extra>"))
        fig3.add_trace(go.Scatter(x=p_df["season_label"], y=p_df["assists_per90"],
            mode="lines+markers", name="Assists/90",
            line=dict(color="#1a50a3", width=2), marker=dict(size=7),
            hovertemplate="%{x}: %{y:.2f}<extra>Assists/90</extra>"))
        fig3.add_trace(go.Scatter(x=p_df["season_label"], y=p_df["availability_rate"],
            mode="lines+markers", name="Availability",
            line=dict(color="#1a7340", width=2, dash="dot"), marker=dict(size=7),
            yaxis="y2",
            hovertemplate="%{x}: %{y:.2f}<extra>Availability</extra>"))
        fig3.update_layout(title="Performance Trends", xaxis_title="Season",
            yaxis=dict(title="Goals / Assists per 90"),
            yaxis2=dict(title="Availability", overlaying="y", side="right", range=[0,1.2]),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5))
        st.plotly_chart(fig3, width='stretch')

    # GRAPH 4 — Sentiment Trend
    with col2:
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            x=p_df["season_label"], y=p_df["vader_compound_score"],
            marker_color=["#1a7340" if v >= 0 else "#b41f1f"
                          for v in p_df["vader_compound_score"]],
            hovertemplate="Season: %{x}<br>Sentiment: %{y:.3f}<extra></extra>",
        ))
        fig4.add_hline(y=0, line_dash="dash", line_color="#888", line_width=1)
        fig4.update_layout(title="Public Sentiment by Season",
            xaxis_title="Season", yaxis_title="Sentiment Score",
            showlegend=False)
        st.plotly_chart(fig4, width='stretch')

# ------------------------------------------------
# Footer
# ------------------------------------------------
st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #1a50a3; /* Match your theme */
        color: black;
        text-align: center;
        padding: 10px;
        z-index: 100;
    }
    </style>
    <div class="footer">
        TransferIQ &nbsp;|&nbsp; Football Player Market Value Prediction &nbsp;|&nbsp;
        Ensemble Model (LSTM + XGBoost) &nbsp;|&nbsp; 2026
    </div>
    """, unsafe_allow_html=True)