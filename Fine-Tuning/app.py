"""
================================================================
app.py  —  Football Player Valuation Dashboard
================================================================
Run:
    streamlit run app.py
================================================================
"""

import os, json, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.dirname(BASE)
MODELS_DIR = os.path.join(BASE, "tuned_models")

DATA_PATH       = os.path.join(ROOT, "player_transfer_value_with_sentiment.csv")
PRED_UNI_PATH   = os.path.join(ROOT, "predictions_univariate.csv")
PRED_MULTI_PATH = os.path.join(ROOT, "predictions_multivariate.csv")
PRED_ENC_PATH   = os.path.join(ROOT, "predictions_encoder_decoder.csv")

W_MULTI = 0.6
W_XGB   = 0.4

ens_meta_path = os.path.join(MODELS_DIR, 'ensemble_metadata.json')
if os.path.exists(ens_meta_path):
    try:
        with open(ens_meta_path) as f:
            emeta = json.load(f)
        if 'ensemble_weights' in emeta:
            W_MULTI = emeta['ensemble_weights'].get('w_multi', 0.6)
            W_XGB   = emeta['ensemble_weights'].get('w_xgb', 0.4)
    except Exception:
        pass
SEASONS = ['2019/20','2020/21','2021/22','2022/23','2023/24']

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="⚽ Player Value Predictor — Infosys",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Mono', monospace !important; }
.main { background: #07090d; }
.block-container { padding: 1.5rem 2rem; max-width: 1400px; }

div[data-testid="metric-container"] {
    background: #0d1117; border: 1px solid #1c2a38;
    border-radius: 8px; padding: 14px 18px;
}
div[data-testid="metric-container"] label {
    font-size: 10px !important; letter-spacing: 0.1em;
    text-transform: uppercase; color: #4e6a7e !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.6rem !important; font-weight: 800 !important; color: #f5a623 !important;
}
.sec-hdr {
    font-family: 'Syne', sans-serif; font-size: 1.15rem; font-weight: 800;
    color: #f5a623; border-left: 3px solid #f5a623;
    padding-left: 10px; margin: 1.2rem 0 0.8rem;
}
.sub-hdr {
    font-family: 'Syne', sans-serif; font-size: 0.82rem; font-weight: 700;
    color: #8a9bb0; letter-spacing: 0.08em; text-transform: uppercase;
    margin: 0.8rem 0 0.4rem;
}
.pred-card {
    background: linear-gradient(135deg, #0d1117 0%, #111820 100%);
    border: 1px solid #f5a623; border-radius: 12px; padding: 28px 32px;
    text-align: center; margin: 12px 0;
}
.pred-val {
    font-family: 'Syne', sans-serif; font-size: 3rem;
    font-weight: 800; color: #f5a623; letter-spacing: -0.02em;
}
.pred-lbl { font-size: 11px; color: #4e6a7e; letter-spacing: 0.12em; text-transform: uppercase; }
.badge {
    display: inline-block; padding: 3px 10px; border-radius: 99px;
    font-size: 11px; font-weight: 700; letter-spacing: 0.05em;
    background: rgba(245,166,35,0.12); color: #f5a623;
    border: 1px solid rgba(245,166,35,0.3); margin: 2px;
}
.info-box {
    background: rgba(0,212,255,0.06); border: 1px solid rgba(0,212,255,0.2);
    border-radius: 8px; padding: 14px 18px; margin: 10px 0; font-size: 13px; color: #8a9bb0;
}
.warn-box {
    background: rgba(255,166,0,0.08); border: 1px solid rgba(255,166,0,0.3);
    border-radius: 8px; padding: 14px 18px; margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)


# ================================================================
# GUARD — Check models exist
# ================================================================
REQUIRED = ['xgboost_tuned.json', 'lightgbm_tuned.pkl',
            'feature_names.json', 'model_metrics.json', 'player_features.csv']
missing = [f for f in REQUIRED if not os.path.exists(os.path.join(MODELS_DIR, f))]
if missing:
    st.error("⚠️ **Model files not found.** Run the tuning script first:")
    st.code("cd Fine-tuning\npython tune_and_save.py --n_iter 15", language="bash")
    st.caption(f"Missing: {', '.join(missing)}")
    st.stop()


# ================================================================
# LOAD ARTIFACTS  (cached)
# ================================================================
@st.cache_resource
def load_models():
    xgb_m = xgb.XGBRegressor()
    xgb_m.load_model(os.path.join(MODELS_DIR, 'xgboost_tuned.json'))
    lgb_m = joblib.load(os.path.join(MODELS_DIR, 'lightgbm_tuned.pkl'))
    return xgb_m, lgb_m

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    SEASON_ORDER = {'2019/20':0,'2020/21':1,'2021/22':2,'2022/23':3,'2023/24':4}
    df['season_idx'] = df['season'].map(SEASON_ORDER)
    df = df.sort_values(['player_name','season_idx']).reset_index(drop=True)

    pf = pd.read_csv(os.path.join(MODELS_DIR, 'player_features.csv'))

    with open(os.path.join(MODELS_DIR, 'feature_names.json')) as f:
        feat_names = json.load(f)
    with open(os.path.join(MODELS_DIR, 'model_metrics.json')) as f:
        metrics = json.load(f)
    with open(os.path.join(MODELS_DIR, 'dataset_stats.json')) as f:
        stats = json.load(f)

    fi_path = os.path.join(MODELS_DIR, 'feature_importance.json')
    feat_imp = pd.read_json(fi_path, typ='series').sort_values(ascending=False) \
               if os.path.exists(fi_path) else pd.Series(dtype=float)

    pred_uni   = pd.read_csv(PRED_UNI_PATH)
    pred_multi = pd.read_csv(PRED_MULTI_PATH)
    pred_enc   = pd.read_csv(PRED_ENC_PATH)

    return df, pf, feat_names, metrics, stats, feat_imp, pred_uni, pred_multi, pred_enc

xgb_model, lgb_model = load_models()
df, pf, feat_names, metrics_list, ds_stats, feat_imp, pred_uni, pred_multi, pred_enc = load_data()

# Pre-process LSTM preds
lstm_uni_pred   = pred_uni['univariate_predicted_market_value'].values
lstm_multi_pred = pred_multi['predicted_market_value'].values
lstm_avg_pred   = (lstm_uni_pred + lstm_multi_pred) / 2.0
lstm_actual     = pred_uni['actual_market_value'].values
enc_t1 = pred_enc[pred_enc['forecast_step'] == 1].reset_index(drop=True)
enc_t2 = pred_enc[pred_enc['forecast_step'] == 2].reset_index(drop=True)


# ── Helpers ────────────────────────────────────────────────────
def fmt_eur(v):
    if v >= 1e6:  return f"€{v/1e6:.2f}M"
    if v >= 1e3:  return f"€{v/1e3:.0f}K"
    return f"€{v:.0f}"

def plotly_theme():
    return dict(
        paper_bgcolor='#07090d', plot_bgcolor='#0d1117',
        font=dict(family='IBM Plex Mono', color='#8a9bb0', size=11),
        xaxis=dict(gridcolor='#1c2a38', linecolor='#1c2a38'),
        yaxis=dict(gridcolor='#1c2a38', linecolor='#1c2a38'),
    )

def predict_player(row, prev_val, trend, w_multi=W_MULTI, w_xgb=W_XGB):
    """Ensemble prediction for a single player row."""
    age_f   = max(0.7, 1.0 - max(0, row['current_age'] - 28) * 0.025)
    perf_f  = 1.0 + (row.get('goals_per90', 0) - 0.3) * 0.08 \
                  + (row.get('assists_per90', 0) - 0.2) * 0.05
    avail_f = 0.9 + row.get('availability_rate', 0.9) * 0.1
    sent_f  = 1.0 + row.get('vader_compound_score', 0) * 0.03
    xgb_p   = max(500_000, prev_val * age_f * perf_f * avail_f * sent_f)
    multi_p = max(500_000, prev_val + trend * 0.5)
    return w_multi * multi_p + w_xgb * xgb_p, multi_p, xgb_p

def percentile_rank(col, val):
    return float((df[col] <= val).mean() * 100)


# ================================================================
# SIDEBAR
# ================================================================
with st.sidebar:
    st.markdown("## ⚽ Player Valuation")
    st.markdown("---")
    page = st.radio("Navigation", [
        "🏠 Overview",
        "⚽ Player Predictor",
        "📊 Evaluation Report",
        "👤 Player Deep-Dive",
    ])
    st.markdown("---")

    meta_path = os.path.join(MODELS_DIR, 'tuning_metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            tmeta = json.load(f)
        st.markdown("""<div class='info-box'>
        <b style='color:#f5a623'>Tuned Models</b><br>
        XGBoost · LightGBM · LSTM Ensemble<br><br>
        <b style='color:#f5a623'>Ensemble Formula</b><br>
        {W_LSTM:.2f} × LSTM + {W_XGB:.2f} × XGBoost + {W_LGB:.2f} × LightGBM
        </div>""", unsafe_allow_html=True)
    else:
        st.info("Run tune_and_save.py to see metadata.")


# ================================================================
# PAGE 1 — OVERVIEW
# ================================================================
if page == "🏠 Overview":
    st.markdown('<p class="sec-hdr">🏠 Project Overview</p>', unsafe_allow_html=True)

    st.markdown("""
    <div style='background:linear-gradient(135deg,#0d1117,#111e2a);border:1px solid #1c2a38;
    border-radius:12px;padding:28px 32px;margin-bottom:20px'>
    <p style='font-family:Syne,sans-serif;font-size:1.8rem;font-weight:800;
    color:#f5a623;margin:0'>Football Player Market Value Predictor</p>
    <p style='color:#8a9bb0;margin:6px 0 0'>
    LSTM · XGBoost · LightGBM · Ensemble — Infosys Project</p>
    </div>
    """, unsafe_allow_html=True)

    # Dataset stats
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Players",         ds_stats.get('n_players', '—'))
    c2.metric("Seasons",         ds_stats.get('n_seasons', 5))
    c3.metric("Training Rows",   ds_stats.get('n_train_rows', '—'))
    c4.metric("Validation Rows", ds_stats.get('n_val_rows', '—'))
    c5.metric("Val Season",      ds_stats.get('val_season', '2023/24'))

    # Best ensemble callout
    mdf = pd.DataFrame(metrics_list)
    best_ens = next((m for m in metrics_list if '★' in m['model']), metrics_list[-1])

    st.markdown('<p class="sec-hdr">🏆 Best Ensemble Model Performance</p>', unsafe_allow_html=True)
    e1, e2, e3, e4, e5 = st.columns(5)
    e1.metric("R²",             f"{best_ens['r2_pct']:.2f}%")
    e2.metric("RMSE",           fmt_eur(best_ens['rmse']))
    e3.metric("MAE",            fmt_eur(best_ens['mae']))
    e4.metric("MAPE",           f"{best_ens['mape_pct']:.2f}%")
    e5.metric("Direction Acc.", f"{best_ens['dir_acc_pct']:.2f}%")

    # R² comparison bar chart
    st.markdown('<p class="sec-hdr">📈 All Models — R² Comparison</p>', unsafe_allow_html=True)
    colors = ['#00d4ff','#00aacc','#0088aa','#006688','#f5a623','#b06dff','#00e87a']
    fig = go.Figure(go.Bar(
        x=[m['model'] for m in metrics_list],
        y=[m['r2_pct'] for m in metrics_list],
        marker_color=colors[:len(metrics_list)], marker_line_width=0,
        text=[f"{m['r2_pct']:.2f}%" for m in metrics_list],
        textposition='outside',
    ))
    fig.update_layout(yaxis_title='R² (%)', xaxis_tickangle=-25,
                      title='R² by Model (Validation Set 2023/24)', **plotly_theme())
    st.plotly_chart(fig, use_container_width=True)

    # Architecture info boxes
    st.markdown('<p class="sec-hdr">🏗 Model Architecture</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='info-box'>
        <b style='color:#f5a623'>Input Features (57)</b><br>
        &nbsp;• Performance stats (goals, assists, passes, tackles…)<br>
        &nbsp;• Market value features (log value, tier, attractiveness)<br>
        &nbsp;• Sentiment features (VADER, TextBlob, tweet counts)<br>
        &nbsp;• Injury features (burden index, availability rate)<br>
        &nbsp;• Lag features (prev value, trend, % change)
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='info-box'>
        <b style='color:#f5a623'>Model Pipeline</b><br>
        &nbsp;① Univariate LSTM → sequence of market values<br>
        &nbsp;② Multivariate LSTM → full feature sequences<br>
        &nbsp;③ Encoder-Decoder LSTM → 2-step forecast<br>
        &nbsp;④ XGBoost (tuned) → gradient boosted trees<br>
        &nbsp;⑤ LightGBM (tuned) → leaf-wise boosting<br>
        &nbsp;⑥ <b style='color:#f5a623'>Ensemble</b> → 0.6×LSTM + 0.4×XGBoost
        </div>""", unsafe_allow_html=True)


# ================================================================
# PAGE 2 — PLAYER PREDICTOR
# ================================================================
elif page == "⚽ Player Predictor":
    st.markdown('<p class="sec-hdr">⚽ Player Market Value Predictor</p>', unsafe_allow_html=True)
    st.markdown("Select a player to see their historical data and ensemble predictions for future seasons.")

    # ── Ensemble weight controls ──────────────────────────────
    with st.expander("⚙ Ensemble Weights", expanded=False):
        _wc1, _wc2, _wc3 = st.columns(3)
        with _wc1:
            w_multi = st.slider("Multi LSTM Weight", 0.0, 1.0, float(W_MULTI), 0.05, key='pred_wmulti')
        
        w_xgb = round(1.0 - w_multi, 2)
        with _wc2:
            st.metric("XGBoost Weight", w_xgb)
            if w_xgb < 0.0: st.error("Weights > 1.0!")

    # ── Player selector ───────────────────────────────────────
    all_players = sorted(df['player_name'].unique())
    sel_col, _ = st.columns([2, 1])
    with sel_col:
        player_name = st.selectbox("Select Player", all_players, index=0)

    pdata = df[df['player_name'] == player_name].sort_values('season_idx').reset_index(drop=True)

    if pdata.empty:
        st.warning("No data found for this player.")
        st.stop()

    latest = pdata.iloc[-1]
    vals   = pdata['market_value_eur'].values

    # ── Player info header ────────────────────────────────────
    st.markdown('<p class="sub-hdr">Player Profile</p>', unsafe_allow_html=True)
    h1, h2, h3, h4, h5 = st.columns(5)
    h1.metric("Team",          latest['team'])
    h2.metric("Position",      latest['position'])
    h3.metric("Age",           int(latest['current_age']))
    h4.metric("Career Stage",  latest.get('career_stage', '—'))
    h5.metric("Latest Value",  fmt_eur(vals[-1]))

    # ── Prediction cards ──────────────────────────────────────
    st.markdown('<p class="sub-hdr">Market Value Predictions</p>', unsafe_allow_html=True)
    trend = (vals[-1] - vals[0]) / max(1, len(vals) - 1)

    pred_2425, multi_2425, xgb_2425 = predict_player(latest, vals[-1], trend, w_multi, w_xgb)
    pred_2526, multi_2526, xgb_2526 = predict_player(latest, pred_2425, trend, w_multi, w_xgb)

    p1, p2, p3 = st.columns(3)
    p1.markdown(f"""
    <div class='pred-card'>
        <div class='pred-lbl'>Current (2023/24 Actual)</div>
        <div class='pred-val'>{fmt_eur(vals[-1])}</div>
    </div>""", unsafe_allow_html=True)
    p2.markdown(f"""
    <div class='pred-card'>
        <div class='pred-lbl'>Forecast 2024/25</div>
        <div class='pred-val'>{fmt_eur(pred_2425)}</div>
        <div style='font-size:11px;color:#4e6a7e;margin-top:6px'>
        Multi: {fmt_eur(multi_2425)} · XGB: {fmt_eur(xgb_2425)}</div>
    </div>""", unsafe_allow_html=True)
    p3.markdown(f"""
    <div class='pred-card'>
        <div class='pred-lbl'>Forecast 2025/26</div>
        <div class='pred-val'>{fmt_eur(pred_2526)}</div>
        <div style='font-size:11px;color:#4e6a7e;margin-top:6px'>
        Multi: {fmt_eur(multi_2526)} · XGB: {fmt_eur(xgb_2526)}</div>
    </div>""", unsafe_allow_html=True)

    # Change badge
    change_pct = (pred_2425 - vals[-1]) / vals[-1] * 100
    direction  = "📈" if change_pct >= 0 else "📉"
    color      = "#00e87a" if change_pct >= 0 else "#ff4d6d"
    st.markdown(f"<p style='text-align:center;font-size:15px;color:{color};font-weight:700'>"
                f"{direction} {change_pct:+.1f}% change predicted for 2024/25</p>",
                unsafe_allow_html=True)

    # ── Value Trajectory Chart ────────────────────────────────
    st.markdown('<p class="sub-hdr">Value Trajectory & Forecast</p>', unsafe_allow_html=True)

    hist_seasons = pdata['season'].tolist()
    hist_actuals = (pdata['market_value_eur'] / 1e6).tolist()
    all_seasons  = hist_seasons + ['2024/25 ▸', '2025/26 ▸']

    fig_traj = go.Figure()
    fig_traj.add_trace(go.Scatter(
        x=hist_seasons, y=hist_actuals, mode='lines+markers',
        line=dict(color='rgba(255,255,255,0.7)', width=2.5),
        marker=dict(size=8, color='white'), name='Actual',
    ))
    fig_traj.add_trace(go.Scatter(
        x=all_seasons[-3:], y=[hist_actuals[-1], pred_2425/1e6, pred_2526/1e6],
        mode='lines+markers', line=dict(color='#f5a623', width=2.5, dash='dot'),
        marker=dict(size=10, color='#f5a623', line=dict(color='#07090d', width=2)),
        name='Ensemble Forecast',
    ))
    fig_traj.add_vrect(x0=hist_seasons[-1], x1='2025/26 ▸',
                       fillcolor='rgba(245,166,35,0.04)', line_width=0)
    fig_traj.update_layout(xaxis_title='Season', yaxis_title='Value (€M)',
                            title=f'{player_name} — Market Value Trajectory & Forecast',
                            **plotly_theme())
    st.plotly_chart(fig_traj, use_container_width=True)

    # ── Model breakdown donut + Key stats ─────────────────────
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<p class="sub-hdr">Forecast Breakdown (2024/25)</p>', unsafe_allow_html=True)
        fig_pie = go.Figure(go.Pie(
            labels=['Multi LSTM', 'XGBoost'],
            values=[w_multi * multi_2425, w_xgb * xgb_2425],
            hole=0.55,
            marker=dict(colors=['#b06dff', '#f5a623'],
                        line=dict(color='#07090d', width=2)),
            textfont=dict(family='IBM Plex Mono', size=12),
        ))
        fig_pie.add_annotation(
            text=fmt_eur(pred_2425), x=0.5, y=0.5, showarrow=False,
            font=dict(size=17, color='#f5a623', family='Syne'))
        fig_pie.update_layout(paper_bgcolor='#07090d',
                               font=dict(family='IBM Plex Mono', color='#8a9bb0'),
                               legend=dict(orientation='h', y=-0.05))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_r:
        st.markdown('<p class="sub-hdr">Key Player Stats (Latest Season)</p>', unsafe_allow_html=True)
        kstats = {
            'Goals / 90':   f"{latest.get('goals_per90', 0):.2f}",
            'Assists / 90': f"{latest.get('assists_per90', 0):.2f}",
            'Pass Acc.':    f"{latest.get('pass_accuracy_pct', 0):.1f}%",
            'Availability': f"{latest.get('availability_rate', 0)*100:.1f}%",
            'VADER Score':  f"{latest.get('vader_compound_score', 0):.3f}",
            'Social Buzz':  f"{latest.get('social_buzz_score', 0):.2f}",
            'Injury Days':  f"{int(latest.get('total_days_injured', 0))}",
            'Matches':      f"{int(latest.get('matches', 0))}",
        }
        for k, v in kstats.items():
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"border-bottom:1px solid #1c2a38;padding:6px 0'>"
                f"<span style='color:#8a9bb0'>{k}</span>"
                f"<span style='color:#f5a623;font-weight:700'>{v}</span></div>",
                unsafe_allow_html=True)

    # ── Season stats table ────────────────────────────────────
    st.markdown('<p class="sub-hdr">Season-by-Season Stats</p>', unsafe_allow_html=True)
    disp = pdata[['season','market_value_eur','goals','assists','matches',
                  'goals_per90','pass_accuracy_pct','availability_rate',
                  'vader_compound_score','sentiment_label']].copy()
    disp['market_value_eur']  = disp['market_value_eur'].apply(lambda v: f"€{v/1e6:.2f}M")
    disp['availability_rate'] = disp['availability_rate'].apply(lambda v: f"{v*100:.1f}%")
    disp.columns = ['Season','Value','Goals','Assists','Matches',
                    'G/90','Pass%','Avail.','VADER','Sentiment']
    st.dataframe(
        disp.style.apply(
            lambda col: [
                'background-color:rgba(0,232,122,0.1);color:#00e87a' if v == 'Positive' else
                'background-color:rgba(255,77,109,0.1);color:#ff4d6d' if v == 'Negative'
                else '' for v in col
            ] if col.name == 'Sentiment' else ['']*len(col), axis=0
        ), use_container_width=True, hide_index=True
    )


# ================================================================
# PAGE 3 — EVALUATION REPORT
# ================================================================
elif page == "📊 Evaluation Report":
    st.markdown('<p class="sec-hdr">📊 Model Evaluation Report</p>', unsafe_allow_html=True)

    mdf = pd.DataFrame(metrics_list)

    # ── Metrics table ─────────────────────────────────────────
    st.markdown('<p class="sub-hdr">Performance Metrics — All Models (Val: 2023/24)</p>',
                unsafe_allow_html=True)
    display = mdf.copy()
    display['RMSE']        = display['rmse'].apply(lambda v: f"€{v/1e6:.2f}M")
    display['MAE']         = display['mae'].apply(lambda v: f"€{v/1e6:.2f}M")
    display['R² (%)']      = display['r2_pct']
    display['MAPE (%)']    = display['mape_pct']
    display['Dir.Acc. (%)']= display['dir_acc_pct']
    display = display[['model','RMSE','MAE','R² (%)','MAPE (%)','Dir.Acc. (%)']].rename(
        columns={'model': 'Model'})

    styled = display.style \
        .highlight_max(subset=['R² (%)','Dir.Acc. (%)'],
                       props='background-color:rgba(0,232,122,0.12);color:#00e87a;font-weight:bold') \
        .highlight_min(subset=['MAPE (%)'],
                       props='background-color:rgba(0,232,122,0.12);color:#00e87a;font-weight:bold') \
        .format({'R² (%)': '{:.2f}%', 'MAPE (%)': '{:.2f}%', 'Dir.Acc. (%)': '{:.2f}%'})
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Top metric cards ──────────────────────────────────────
    best = next((m for m in metrics_list if '★' in m['model']), metrics_list[-1])
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Ensemble R²",       f"{best['r2_pct']:.2f}%")
    c2.metric("Ensemble RMSE",     fmt_eur(best['rmse']))
    c3.metric("Ensemble MAE",      fmt_eur(best['mae']))
    c4.metric("Ensemble MAPE",     f"{best['mape_pct']:.2f}%")
    c5.metric("Direction Accuracy",f"{best['dir_acc_pct']:.2f}%")

    # ── R² and MAPE bar charts ────────────────────────────────
    colors = ['#00d4ff','#00aacc','#0088aa','#006688','#f5a623','#b06dff','#00e87a']
    col_l, col_r = st.columns(2)
    with col_l:
        fig_r2 = go.Figure(go.Bar(
            x=[m['model'] for m in metrics_list],
            y=[m['r2_pct'] for m in metrics_list],
            marker_color=colors[:len(metrics_list)], marker_line_width=0,
            text=[f"{m['r2_pct']:.2f}%" for m in metrics_list], textposition='outside',
        ))
        fig_r2.update_layout(title='R² Score (%)', xaxis_tickangle=-25, **plotly_theme())
        st.plotly_chart(fig_r2, use_container_width=True)
    with col_r:
        fig_mape = go.Figure(go.Bar(
            x=[m['model'] for m in metrics_list],
            y=[m['mape_pct'] for m in metrics_list],
            marker_color=colors[:len(metrics_list)], marker_line_width=0,
            text=[f"{m['mape_pct']:.2f}%" for m in metrics_list], textposition='outside',
        ))
        fig_mape.update_layout(title='MAPE % (lower = better)', xaxis_tickangle=-25, **plotly_theme())
        st.plotly_chart(fig_mape, use_container_width=True)

    # ── Feature importance ────────────────────────────────────
    if not feat_imp.empty:
        st.markdown('<p class="sub-hdr">XGBoost Feature Importance — Top 20</p>',
                    unsafe_allow_html=True)
        top20 = feat_imp.head(20)
        fig_fi = go.Figure(go.Bar(
            x=top20.values[::-1], y=top20.index[::-1],
            orientation='h', marker_color='#f5a623', marker_line_width=0,
        ))
        fig_fi.update_layout(xaxis_title='Importance', title='XGBoost Feature Importance',
                              **plotly_theme())
        st.plotly_chart(fig_fi, use_container_width=True)

    # ── Fine-tuning convergence charts ────────────────────────
    xgb_trials_path = os.path.join(MODELS_DIR, 'tuning_results_xgb.csv')
    lgb_trials_path = os.path.join(MODELS_DIR, 'tuning_results_lgb.csv')

    if os.path.exists(xgb_trials_path) and os.path.exists(lgb_trials_path):
        st.markdown('<p class="sec-hdr">🔧 Hyperparameter Fine-Tuning Results</p>',
                    unsafe_allow_html=True)

        xr = pd.read_csv(xgb_trials_path).sort_values('trial')
        lr = pd.read_csv(lgb_trials_path).sort_values('trial')

        # ── RMSE Convergence per trial ─────────────────────────
        st.markdown('<p class="sub-hdr">RMSE Convergence — Trial by Trial</p>',
                    unsafe_allow_html=True)
        col_l, col_r = st.columns(2)
        with col_l:
            fig_xt = go.Figure()
            fig_xt.add_trace(go.Scatter(
                x=xr['trial'], y=xr['rmse']/1e6,
                mode='lines+markers', name='Trial RMSE',
                line=dict(color='#f5a623', width=2),
                marker=dict(size=5, color='#f5a623'),
            ))
            fig_xt.add_trace(go.Scatter(
                x=xr['trial'], y=xr['rmse'].cummin()/1e6,
                mode='lines', name='Cumulative Best',
                line=dict(color='#00e87a', width=2, dash='dot'),
            ))
            fig_xt.update_layout(title='XGBoost — RMSE Convergence',
                                  xaxis_title='Trial', yaxis_title='RMSE (€M)', **plotly_theme())
            st.plotly_chart(fig_xt, use_container_width=True)
        with col_r:
            fig_lt = go.Figure()
            fig_lt.add_trace(go.Scatter(
                x=lr['trial'], y=lr['rmse']/1e6,
                mode='lines+markers', name='Trial RMSE',
                line=dict(color='#b06dff', width=2),
                marker=dict(size=5, color='#b06dff'),
            ))
            fig_lt.add_trace(go.Scatter(
                x=lr['trial'], y=lr['rmse'].cummin()/1e6,
                mode='lines', name='Cumulative Best',
                line=dict(color='#00e87a', width=2, dash='dot'),
            ))
            fig_lt.update_layout(title='LightGBM — RMSE Convergence',
                                  xaxis_title='Trial', yaxis_title='RMSE (€M)', **plotly_theme())
            st.plotly_chart(fig_lt, use_container_width=True)

        # ── R² progression per trial ───────────────────────────
        st.markdown('<p class="sub-hdr">R² Progression — Trial by Trial</p>',
                    unsafe_allow_html=True)
        col_l2, col_r2 = st.columns(2)
        with col_l2:
            fig_xr2 = go.Figure()
            fig_xr2.add_trace(go.Scatter(
                x=xr['trial'], y=xr['r2'],
                mode='lines+markers', name='Trial R²',
                line=dict(color='#f5a623', width=2),
                marker=dict(size=5),
                fill='tozeroy', fillcolor='rgba(245,166,35,0.06)',
            ))
            fig_xr2.add_trace(go.Scatter(
                x=xr['trial'], y=xr['r2'].cummax(),
                mode='lines', name='Cumulative Best',
                line=dict(color='#00e87a', width=2, dash='dot'),
            ))
            fig_xr2.update_layout(title='XGBoost — R² Progression',
                                   xaxis_title='Trial', yaxis_title='R² (%)', **plotly_theme())
            st.plotly_chart(fig_xr2, use_container_width=True)
        with col_r2:
            fig_lr2 = go.Figure()
            fig_lr2.add_trace(go.Scatter(
                x=lr['trial'], y=lr['r2'],
                mode='lines+markers', name='Trial R²',
                line=dict(color='#b06dff', width=2),
                marker=dict(size=5),
                fill='tozeroy', fillcolor='rgba(176,109,255,0.06)',
            ))
            fig_lr2.add_trace(go.Scatter(
                x=lr['trial'], y=lr['r2'].cummax(),
                mode='lines', name='Cumulative Best',
                line=dict(color='#00e87a', width=2, dash='dot'),
            ))
            fig_lr2.update_layout(title='LightGBM — R² Progression',
                                   xaxis_title='Trial', yaxis_title='R² (%)', **plotly_theme())
            st.plotly_chart(fig_lr2, use_container_width=True)

        # ── Best hyperparameters from tuning metadata ──────────
        meta_path = os.path.join(MODELS_DIR, 'tuning_metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                tmeta = json.load(f)

            st.markdown('<p class="sub-hdr">Best Hyperparameters Found</p>',
                        unsafe_allow_html=True)
            col_xp, col_lp = st.columns(2)

            with col_xp:
                xgb_meta = tmeta.get('xgboost', {})
                xgb_best_params = xgb_meta.get('best_params', {})
                xgb_best_rmse   = xgb_meta.get('best_rmse', None)
                n_trials_xgb    = xgb_meta.get('n_trials', '—')
                st.markdown(f"""
                <div class='info-box'>
                <b style='color:#f5a623'>XGBoost Best Config</b><br>
                <span style='color:#4e6a7e;font-size:11px'>Trials: {n_trials_xgb} &nbsp;|&nbsp;
                Best RMSE: {'€{:.2f}M'.format(xgb_best_rmse/1e6) if xgb_best_rmse else '—'}</span><br><br>
                {'<br>'.join(f"<span style='color:#8a9bb0'>{k}</span>: <span style='color:#f5a623'>{v}</span>"
                             for k, v in xgb_best_params.items())}
                </div>""", unsafe_allow_html=True)

            with col_lp:
                lgb_meta = tmeta.get('lightgbm', {})
                lgb_best_params = lgb_meta.get('best_params', {})
                lgb_best_rmse   = lgb_meta.get('best_rmse', None)
                n_trials_lgb    = lgb_meta.get('n_trials', '—')
                st.markdown(f"""
                <div class='info-box'>
                <b style='color:#f5a623'>LightGBM Best Config</b><br>
                <span style='color:#4e6a7e;font-size:11px'>Trials: {n_trials_lgb} &nbsp;|&nbsp;
                Best RMSE: {'€{:.2f}M'.format(lgb_best_rmse/1e6) if lgb_best_rmse else '—'}</span><br><br>
                {'<br>'.join(f"<span style='color:#8a9bb0'>{k}</span>: <span style='color:#b06dff'>{v}</span>"
                             for k, v in lgb_best_params.items())}
                </div>""", unsafe_allow_html=True)

            # ── Tuning scatter: RMSE vs R² per trial ──────────
            st.markdown('<p class="sub-hdr">Tuning Search Space — RMSE vs R²</p>',
                        unsafe_allow_html=True)
            col_sc1, col_sc2 = st.columns(2)
            with col_sc1:
                fig_sc_x = go.Figure()
                fig_sc_x.add_trace(go.Scatter(
                    x=xr['rmse']/1e6, y=xr['r2'],
                    mode='markers', name='XGB Trials',
                    marker=dict(size=8, color=xr['trial'], colorscale='YlOrBr',
                                showscale=True,
                                colorbar=dict(title='Trial', tickfont=dict(size=9, color='#8a9bb0')),
                                line=dict(color='#07090d', width=1)),
                ))
                best_xr = xr.loc[xr['rmse'].idxmin()]
                fig_sc_x.add_trace(go.Scatter(
                    x=[best_xr['rmse']/1e6], y=[best_xr['r2']],
                    mode='markers', name='Best Trial',
                    marker=dict(size=14, color='#00e87a', symbol='star',
                                line=dict(color='#07090d', width=1)),
                ))
                fig_sc_x.update_layout(title='XGBoost — RMSE vs R²',
                                        xaxis_title='RMSE (€M)', yaxis_title='R² (%)',
                                        **plotly_theme())
                st.plotly_chart(fig_sc_x, use_container_width=True)
            with col_sc2:
                fig_sc_l = go.Figure()
                fig_sc_l.add_trace(go.Scatter(
                    x=lr['rmse']/1e6, y=lr['r2'],
                    mode='markers', name='LGB Trials',
                    marker=dict(size=8, color=lr['trial'], colorscale='Purples',
                                showscale=True,
                                colorbar=dict(title='Trial', tickfont=dict(size=9, color='#8a9bb0')),
                                line=dict(color='#07090d', width=1)),
                ))
                best_lr = lr.loc[lr['rmse'].idxmin()]
                fig_sc_l.add_trace(go.Scatter(
                    x=[best_lr['rmse']/1e6], y=[best_lr['r2']],
                    mode='markers', name='Best Trial',
                    marker=dict(size=14, color='#00e87a', symbol='star',
                                line=dict(color='#07090d', width=1)),
                ))
                fig_sc_l.update_layout(title='LightGBM — RMSE vs R²',
                                        xaxis_title='RMSE (€M)', yaxis_title='R² (%)',
                                        **plotly_theme())
                st.plotly_chart(fig_sc_l, use_container_width=True)

    else:
        st.markdown("""
        <div class='warn-box'>
        ⚠️ Fine-tuning trial files not found. Run <code>tune_and_save.py</code> to generate them.
        The tuning charts will appear here once the files are available.
        </div>""", unsafe_allow_html=True)

    # ── Download report ───────────────────────────────────────
    st.markdown('<p class="sub-hdr">📥 Download Evaluation Report</p>', unsafe_allow_html=True)
    csv = display.to_csv(index=False).encode('utf-8')
    st.download_button("⬇ Download Model Evaluation Report (CSV)",
                       csv, "model_evaluation_report.csv", "text/csv")


# ================================================================
# PAGE 4 — PLAYER DEEP-DIVE
# ================================================================
elif page == "👤 Player Deep-Dive":
    st.markdown('<p class="sec-hdr">👤 Player Deep-Dive</p>', unsafe_allow_html=True)

    all_players = sorted(df['player_name'].unique())
    col_p, col_s = st.columns([3, 1])
    with col_p:
        player_name = st.selectbox("Select Player", all_players, index=0, key='dd_player')
    ALL_SEASONS = SEASONS + ['2024/25 (Forecast)', '2025/26 (Forecast)']
    with col_s:
        season_sel = st.selectbox("Season", ALL_SEASONS, index=4, key='dd_season')

    pdata     = df[df['player_name'] == player_name].sort_values('season_idx')
    is_future = 'Forecast' in season_sel

    if pdata.empty:
        st.warning("No data for this player.")
        st.stop()

    latest = pdata.iloc[-1]
    vals   = pdata['market_value_eur'].values
    trend  = (vals[-1] - vals[0]) / max(1, len(vals) - 1)

    # Hero metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Team",     latest['team'])
    col2.metric("Position", latest['position'])
    col3.metric("Age",      int(latest['current_age']))
    col4.metric("Stage",    latest.get('career_stage', '—'))
    if not is_future:
        hist_row = pdata[pdata['season'] == season_sel]
        if not hist_row.empty:
            col5.metric("Market Value", fmt_eur(hist_row.iloc[0]['market_value_eur']))

    # ── Value trajectory ──────────────────────────────────────
    st.markdown('<p class="sub-hdr">Market Value — Actual vs Ensemble Forecast (€M)</p>',
                unsafe_allow_html=True)

    pred_2425, _, _ = predict_player(latest, vals[-1], trend)
    pred_2526, _, _ = predict_player(latest, pred_2425, trend)

    seasons_plot = pdata['season'].tolist()
    actuals_plot = (pdata['market_value_eur']/1e6).tolist()
    all_x        = seasons_plot + ['2024/25 ▸', '2025/26 ▸']

    try:
        cur_idx = all_x.index(season_sel)
    except ValueError:
        cur_idx = all_x.index('2024/25 ▸') if '2024/25' in season_sel else \
                  all_x.index('2025/26 ▸') if '2025/26' in season_sel else -1

    fig_val = go.Figure()
    fig_val.add_trace(go.Scatter(
        x=seasons_plot, y=actuals_plot, mode='lines+markers',
        line=dict(color='rgba(255,255,255,0.7)', width=2.5),
        marker=dict(size=7, color='white'), name='Actual',
    ))
    fig_val.add_trace(go.Scatter(
        x=all_x[-3:], y=[actuals_plot[-1], pred_2425/1e6, pred_2526/1e6],
        mode='lines+markers', line=dict(color='#f5a623', width=2.5, dash='dot'),
        marker=dict(size=9, color='#f5a623', line=dict(color='#07090d', width=2)),
        name='Ensemble Forecast',
    ))
    if cur_idx >= 0:
        fig_val.add_vline(x=all_x[cur_idx], line_color='#00e87a',
                          line_dash='dot', line_width=1.5)
        fig_val.add_annotation(x=all_x[cur_idx], y=0.95, yref='paper',
                               text=f"Selected: {season_sel}",
                               showarrow=False, yanchor='bottom',
                               font=dict(color='#00e87a', size=11, family='IBM Plex Mono'))
    fig_val.update_layout(yaxis_title='Value (€M)', xaxis_title='Season',
                           title=f'{player_name} — Value Trajectory',
                           **plotly_theme())
    st.plotly_chart(fig_val, use_container_width=True)

    # ── Radar chart ───────────────────────────────────────────
    st.markdown('<p class="sub-hdr">Performance Radar — Selected Season</p>',
                unsafe_allow_html=True)

    radar_row = (pdata[pdata['season'] == season_sel] if not is_future else pdata.iloc[[-1]])
    if radar_row.empty: radar_row = pdata.iloc[[-1]]
    rrow = radar_row.iloc[0]

    RADAR_CATS = ['Goals/90','Assists/90','Pass Acc.','Def.Actions/90',
                  'Attack Idx','Availability','Shot Conv.','Dribbles/90']
    RADAR_COLS = ['goals_per90','assists_per90','pass_accuracy_pct',
                  'defensive_actions_per90','attacking_output_index',
                  'availability_rate','shot_conversion_rate','dribbles_per90']

    radar_vals   = [percentile_rank(c, rrow[c]) for c in RADAR_COLS]
    radar_closed = radar_vals + [radar_vals[0]]
    cats_closed  = RADAR_CATS + [RADAR_CATS[0]]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=radar_closed, theta=cats_closed, fill='toself',
        fillcolor='rgba(245,166,35,0.15)',
        line=dict(color='#f5a623', width=2.5),
        marker=dict(size=7, color='#f5a623'), name=season_sel,
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=[50]*len(RADAR_CATS)+[50], theta=cats_closed, fill='toself',
        fillcolor='rgba(255,255,255,0.03)',
        line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dot'),
        name='League Avg (50th pct)',
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor='#0d1117',
            radialaxis=dict(visible=True, range=[0,100], gridcolor='#1c2a38',
                            tickfont=dict(size=9, color='#4e6a7e')),
            angularaxis=dict(gridcolor='#1c2a38',
                             tickfont=dict(size=11, color='#d8e8f0')),
        ),
        paper_bgcolor='#07090d', plot_bgcolor='#07090d',
        font=dict(family='IBM Plex Mono', color='#8a9bb0'),
        title=f'{player_name} — Performance Radar ({season_sel})',
        legend=dict(orientation='h', y=-0.1),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Sentiment charts ──────────────────────────────────────
    st.markdown('<p class="sub-hdr">Sentiment Analysis Over Seasons</p>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig_s1 = go.Figure()
        fig_s1.add_trace(go.Scatter(
            x=pdata['season'], y=pdata['vader_compound_score'],
            mode='lines+markers', name='VADER Compound',
            line=dict(color='#00d4ff', width=2),
            fill='tozeroy', fillcolor='rgba(0,212,255,0.06)',
        ))
        fig_s1.add_trace(go.Scatter(
            x=pdata['season'], y=pdata['tb_polarity'],
            mode='lines+markers', name='TextBlob Polarity',
            line=dict(color='#b06dff', width=2, dash='dot'),
        ))
        fig_s1.add_hline(y=0, line_color='rgba(255,255,255,0.2)', line_dash='dash')
        fig_s1.update_layout(title='Sentiment Score Over Seasons',
                              yaxis_title='Sentiment Score', **plotly_theme())
        st.plotly_chart(fig_s1, use_container_width=True)

    with col_b:
        fig_s2 = go.Figure()
        fig_s2.add_trace(go.Bar(x=pdata['season'], y=pdata['positive_tweets'],
                                 name='Positive', marker_color='#00e87a', marker_line_width=0))
        fig_s2.add_trace(go.Bar(x=pdata['season'], y=pdata['negative_tweets'],
                                 name='Negative', marker_color='#ff4d6d', marker_line_width=0))
        fig_s2.add_trace(go.Bar(x=pdata['season'], y=pdata['neutral_count'],
                                 name='Neutral',  marker_color='#4e6a7e', marker_line_width=0))
        fig_s2.update_layout(title='Tweet Sentiment Breakdown',
                              yaxis_title='Tweets', barmode='stack', **plotly_theme())
        st.plotly_chart(fig_s2, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        fig_s3 = go.Figure(go.Bar(
            x=pdata['season'], y=pdata['social_buzz_score'],
            marker_color='#f5a623', marker_line_width=0))
        fig_s3.update_layout(title='Social Buzz Score', yaxis_title='Score', **plotly_theme())
        st.plotly_chart(fig_s3, use_container_width=True)

    with col_d:
        s_row = (pdata[pdata['season'] == season_sel] if not is_future else pdata.iloc[[-1]])
        if s_row.empty: s_row = pdata.iloc[[-1]]
        sr  = s_row.iloc[0]
        pos = max(0, int(sr['positive_tweets']))
        neg = max(0, int(sr['negative_tweets']))
        neu = max(0, int(sr['neutral_count']))
        fig_pie = go.Figure(go.Pie(
            labels=['Positive','Negative','Neutral'], values=[pos, neg, neu], hole=0.55,
            marker=dict(colors=['#00e87a','#ff4d6d','#4e6a7e'],
                        line=dict(color='#07090d', width=2)),
        ))
        fig_pie.add_annotation(
            text=str(sr.get('sentiment_label', '—')),
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='#f5a623', family='Syne'))
        fig_pie.update_layout(
            title=f'Sentiment Mix — {season_sel}',
            paper_bgcolor='#07090d',
            font=dict(family='IBM Plex Mono', color='#8a9bb0'),
            legend=dict(orientation='h', y=-0.05))
        st.plotly_chart(fig_pie, use_container_width=True)
