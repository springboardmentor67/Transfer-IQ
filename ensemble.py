import numpy as np
import pandas as pd
import pickle, os, json, warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")
np.random.seed(42)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("xgboost not found. Install with: pip install xgboost")

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "player_transfer_value_with_sentiment.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR  = os.path.join(MODELS_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

class LSTMCell:
    def __init__(self, input_size, hidden_size):
        scale = 0.1
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.bf = np.zeros((hidden_size, 1))
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.bi = np.zeros((hidden_size, 1))
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.bc = np.zeros((hidden_size, 1))
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.bo = np.zeros((hidden_size, 1))
        self.hidden_size = hidden_size

    @staticmethod
    def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -15, 15)))
    @staticmethod
    def tanh(x):    return np.tanh(np.clip(x, -15, 15))

    def forward(self, x, h_prev, c_prev):
        combined = np.vstack([h_prev, x])
        f = self.sigmoid(self.Wf @ combined + self.bf)
        i = self.sigmoid(self.Wi @ combined + self.bi)
        c_tilde = self.tanh(self.Wc @ combined + self.bc)
        c = f * c_prev + i * c_tilde
        o = self.sigmoid(self.Wo @ combined + self.bo)
        h = o * self.tanh(c)
        return h, c

    def run_sequence(self, X_seq):
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        for t in range(X_seq.shape[0]):
            h, c = self.forward(X_seq[t].reshape(-1, 1), h, c)
        return h, c


class SimpleLSTM:
    def __init__(self, input_size, hidden_size=32, output_size=1):
        self.cell = LSTMCell(input_size, hidden_size)
        self.Wy   = np.random.randn(output_size, hidden_size) * 0.1
        self.by   = np.zeros((output_size, 1))
        self.hidden_size  = hidden_size
        self.input_size   = input_size
        self.train_losses = []
        self.val_losses   = []

    def predict_one(self, X_seq):
        h, _ = self.cell.run_sequence(X_seq)
        return float((self.Wy @ h + self.by).flatten()[0])

    def predict_batch(self, X):
        return np.array([self.predict_one(X[i]) for i in range(len(X))])

    def _get_params(self):
        return [self.cell.Wf, self.cell.bf, self.cell.Wi, self.cell.bi,
                self.cell.Wc, self.cell.bc, self.cell.Wo, self.cell.bo,
                self.Wy, self.by]

    def _set_params(self, params):
        for k, p in zip(["Wf","bf","Wi","bi","Wc","bc","Wo","bo"], params[:8]):
            setattr(self.cell, k, p.copy())
        self.Wy = params[8].copy()
        self.by = params[9].copy()


class EncoderDecoderLSTM:
    def __init__(self, input_size, hidden_size=32, dec_steps=2):
        self.encoder   = LSTMCell(input_size, hidden_size)
        self.decoder   = LSTMCell(1, hidden_size)
        self.Wy        = np.random.randn(1, hidden_size) * 0.1
        self.by        = np.zeros((1, 1))
        self.dec_steps = dec_steps
        self.hidden_size  = hidden_size
        self.train_losses = []
        self.val_losses   = []

    def predict_one(self, X_seq):
        h, c = self.encoder.run_sequence(X_seq)
        outputs   = []
        dec_input = X_seq[-1, 0:1].reshape(-1, 1)
        for _ in range(self.dec_steps):
            h, c  = self.decoder.forward(dec_input, h, c)
            out   = float((self.Wy @ h + self.by).flatten()[0])
            outputs.append(out)
            dec_input = np.array([[out]])
        return np.array(outputs)

    def predict_batch(self, X):
        return np.array([self.predict_one(X[i]) for i in range(len(X))])


def ld(fname):
    return pickle.load(open(os.path.join(MODELS_DIR, fname), "rb"))

print("="*60)
print("Loading pre-trained LSTM models...")
print("="*60)

model_mv    = ld("model_multivariate.pkl")
model_uni   = ld("model_univariate.pkl")
model_ed    = ld("model_encoder_decoder.pkl")
mv_scalers  = ld("scalers_multivariate.pkl")
uni_scaler  = ld("scaler_univariate.pkl")
mv_features = ld("mv_features.pkl")
uni_data    = ld("uni_data.pkl")
uni_data_scaled = ld("uni_data_scaled.pkl")
player_list = ld("player_list.pkl")

print(f"✅ Loaded all 3 LSTM models | {len(player_list)} players | {len(mv_features)} features")

df = pd.read_csv(DATA_PATH)
SEASONS = ["2019/20", "2020/21", "2021/22", "2022/23", "2023/24"]
df["season_idx"] = df["season"].map({s: i for i, s in enumerate(SEASONS)})

counts      = df.groupby("player_name")["season"].count()
full_players = counts[counts == 5].index.tolist()
df          = df[df["player_name"].isin(full_players)].copy()
df          = df.sort_values(["player_name", "season_idx"]).reset_index(drop=True)

print(f"✅ Dataset loaded: {len(df)} rows | {len(df.columns)} columns")

n_players = len(player_list)
n_seasons = 5
n_feats   = len(mv_features)

mv_matrix = np.zeros((n_players, n_seasons, n_feats))
for pi, pname in enumerate(player_list):
    pdata = df[df["player_name"] == pname].sort_values("season_idx")
    for si in range(min(len(pdata), n_seasons)):
        mv_matrix[pi, si, :] = pdata.iloc[si][mv_features].values.astype(float)

mv_matrix_scaled = mv_matrix.copy()
for fi in range(n_feats):
    flat = mv_matrix[:, :, fi].reshape(-1, 1)
    mv_matrix_scaled[:, :, fi] = mv_scalers[fi].transform(flat).reshape(n_players, n_seasons)

# Univariate 3D matrix (already saved as uni_data_scaled)
uni_3d = uni_data_scaled[:, :, np.newaxis]   # (n_players, 5, 1)

print("\n" + "="*60)
print("Generating all LSTM predictions for all players...")
print("="*60)

SEQ_LEN = 3

# Dictionaries to hold predictions per (player_name, season_idx)
pred_mv_dict  = {}
pred_uni_dict = {}
pred_ed1_dict = {}
pred_ed2_dict = {}

for pi, pname in enumerate(player_list):
    for target_season in range(SEQ_LEN, n_seasons):

        # ── Multivariate LSTM ──
        mv_seq      = mv_matrix_scaled[pi, target_season - SEQ_LEN : target_season, :]
        mv_scaled   = model_mv.predict_one(mv_seq)
        mv_eur      = float(mv_scalers[0].inverse_transform([[mv_scaled]])[0][0])
        pred_mv_dict[(pname, target_season)] = mv_eur

        # ── Univariate LSTM ──
        uni_seq     = uni_3d[pi, target_season - SEQ_LEN : target_season, :]
        uni_scaled  = model_uni.predict_one(uni_seq)
        dummy       = np.zeros((1, uni_data.shape[1]))
        dummy[0, 0] = uni_scaled
        uni_eur     = float(uni_scaler.inverse_transform(dummy)[0, 0])
        pred_uni_dict[(pname, target_season)] = uni_eur

        # ── Encoder-Decoder LSTM (Step 1 & Step 2) ──
        ed_seq      = uni_3d[pi, target_season - SEQ_LEN : target_season, :]
        ed_preds    = model_ed.predict_one(ed_seq)          # (2,)
        dummy2      = np.zeros((2, uni_data.shape[1]))
        dummy2[:, 0] = ed_preds
        ed_orig     = uni_scaler.inverse_transform(dummy2)[:, 0]
        pred_ed1_dict[(pname, target_season)] = float(ed_orig[0])
        pred_ed2_dict[(pname, target_season)] = float(ed_orig[1])

print(f"✅ Generated {len(pred_mv_dict) * 4} total LSTM predictions ({len(pred_mv_dict)} rows × 4 columns)")

# ── Map all 4 predictions back onto the dataframe ────────────────────────────
df["lstm_mv_predicted_value"]  = df.apply(lambda r: pred_mv_dict .get((r["player_name"], r["season_idx"]), np.nan), axis=1)
df["lstm_uni_predicted_value"] = df.apply(lambda r: pred_uni_dict.get((r["player_name"], r["season_idx"]), np.nan), axis=1)
df["lstm_ed_step1_value"]      = df.apply(lambda r: pred_ed1_dict.get((r["player_name"], r["season_idx"]), np.nan), axis=1)
df["lstm_ed_step2_value"]      = df.apply(lambda r: pred_ed2_dict.get((r["player_name"], r["season_idx"]), np.nan), axis=1)

NEW_LSTM_COLS = [
    "lstm_mv_predicted_value",
    "lstm_uni_predicted_value",
    "lstm_ed_step1_value",
    "lstm_ed_step2_value",
]

NEW_CSV_PATH  = os.path.join(BASE_DIR, "player_transfer_value_ensemble.csv")
df.to_csv(NEW_CSV_PATH, index=False)

original_cols = len(df.columns) - len(NEW_LSTM_COLS) - 1  # subtract new cols + season_idx
print(f"\n✅ New dataset saved: player_transfer_value_ensemble.csv")
print(f"   Original columns  : {original_cols}")
print(f"   New columns added : {NEW_LSTM_COLS}")
print(f"   Total columns     : {len(df.columns)}")
print(f"   Total rows        : {len(df)}")


# Only keep rows that have all LSTM predictions (season_idx >= 3)
df_xgb = df.dropna(subset=NEW_LSTM_COLS).copy()

FEATURE_COLS = mv_features + NEW_LSTM_COLS   # 14 + 4 = 18 features
TARGET_COL   = "market_value_eur"

X = df_xgb[FEATURE_COLS].values.astype(float)
y = df_xgb[TARGET_COL].values.astype(float)

# 80/20 train-validation split
split      = int(0.8 * len(X))
X_tr, X_vl = X[:split], X[split:]
y_tr, y_vl = y[:split], y[split:]

print(f"\n✅ XGBoost dataset ready")
print(f"   Features : {len(FEATURE_COLS)} ({len(mv_features)} original + {len(NEW_LSTM_COLS)} LSTM predictions)")
print(f"   Train    : {len(X_tr)} samples")
print(f"   Val      : {len(X_vl)} samples")

results = {}

lstm_val_preds  = df_xgb["lstm_mv_predicted_value"].values[split:]
lstm_actual     = y_vl

rmse_lstm = float(np.sqrt(mean_squared_error(lstm_actual, lstm_val_preds)))
mae_lstm  = float(mean_absolute_error(lstm_actual, lstm_val_preds))
ss_res    = np.sum((lstm_actual - lstm_val_preds) ** 2)
ss_tot    = np.sum((lstm_actual - np.mean(lstm_actual)) ** 2)
r2_lstm   = float(max(0, 1 - ss_res / ss_tot))

results["Multivariate LSTM (baseline)"] = {
    "RMSE": rmse_lstm, "MAE": mae_lstm, "R2": r2_lstm
}

print(f"\n{'='*60}")
print("Model Performance on Validation Set")
print(f"{'='*60}")
print(f"  {'Multivariate LSTM (baseline)':<35} RMSE: €{rmse_lstm/1e6:.3f}M  MAE: €{mae_lstm/1e6:.3f}M  R²: {r2_lstm:.4f}")

xgb_model       = None
xgb_best_params = {}
xgb_top_runs    = []

# ── Cross-validation splitter (shared) ───────────────────────────────────────
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# ── Helper: extract top N CV runs from GridSearch/RandomSearch results ────────
def top_cv_runs(cv_results, n=10):
    df_cv = pd.DataFrame(cv_results)
    df_cv["rmse"] = -df_cv["mean_test_score"]
    df_cv = df_cv.sort_values("rmse").head(n)
    rows = []
    for _, row in df_cv.iterrows():
        entry = {"rmse": round(float(row["rmse"]), 2),
                 "std":  round(float(row["std_test_score"]), 2)}
        for col in df_cv.columns:
            if col.startswith("param_"):
                val = row[col]
                entry[col.replace("param_", "")] = None if (hasattr(val, '__class__') and val.__class__.__name__ == 'MaskedConstant') else val
        rows.append(entry)
    return rows

XGB_PARAM_GRID = {
    "max_depth":     [3, 5, 7],
    "n_estimators":  [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample":     [0.8, 1.0],
}

if HAS_XGB:
    n_combos = 1
    for v in XGB_PARAM_GRID.values(): n_combos *= len(v)
    print(f"\nXGBoost GridSearchCV  ({n_combos} combos × 5 folds = {n_combos*5} fits)...")

    base_xgb = xgb.XGBRegressor(
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
        objective="reg:squarederror",
    )

    grid_xgb = GridSearchCV(
        estimator=base_xgb,
        param_grid=XGB_PARAM_GRID,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    grid_xgb.fit(X, y)

    xgb_model       = grid_xgb.best_estimator_
    xgb_best_params = grid_xgb.best_params_
    xgb_cv_rmse     = round(float(-grid_xgb.best_score_), 2)
    xgb_top_runs    = top_cv_runs(grid_xgb.cv_results_)

    print(f"  Best params : {xgb_best_params}")
    print(f"  CV RMSE     : €{xgb_cv_rmse/1e6:.3f}M")

    xgb_val_preds = xgb_model.predict(X_vl)
    rmse_xgb = float(np.sqrt(mean_squared_error(y_vl, xgb_val_preds)))
    mae_xgb  = float(mean_absolute_error(y_vl, xgb_val_preds))
    ss_res   = np.sum((y_vl - xgb_val_preds) ** 2)
    r2_xgb   = float(max(0, 1 - ss_res / ss_tot))

    results["XGBoost + LSTM Ensemble"] = {
        "RMSE": rmse_xgb, "MAE": mae_xgb, "R2": r2_xgb
    }
    print(f"  {'XGBoost + LSTM Ensemble':<35} RMSE: €{rmse_xgb/1e6:.3f}M  MAE: €{mae_xgb/1e6:.3f}M  R²: {r2_xgb:.4f}")

if xgb_model:
    pickle.dump(xgb_model, open(os.path.join(MODELS_DIR, "model_xgb_ensemble.pkl"), "wb"))
    print(f"\n✅ XGBoost tuned model saved → models/model_xgb_ensemble.pkl")

# Save feature list so app.py knows what columns to use
pickle.dump(FEATURE_COLS, open(os.path.join(MODELS_DIR, "ensemble_feature_cols.pkl"), "wb"))

print("\n" + "="*70)
print("HYPERPARAMETER TUNING & FINAL ENSEMBLE RESULTS")
print("="*70)
print(f"{'Model':<38} {'RMSE':>10} {'MAE':>10} {'R²':>8}")
print("-"*70)
for name, m in results.items():
    tag = " ★" if name == max(results, key=lambda k: results[k]["R2"]) else ""
    print(f"  {name:<36} €{m['RMSE']/1e6:>6.3f}M  €{m['MAE']/1e6:>6.3f}M  {m['R2']:>7.4f}{tag}")
print("="*70)
print("★ = Best R²")
if HAS_XGB:
    print(f"\n  XGBoost best params (GridSearchCV): {xgb_best_params}")

SAMPLE = 120

def sample_arrays(actual, predicted, n=SAMPLE):
    actual    = np.array(actual)
    predicted = np.array(predicted)
    if len(actual) <= n:
        return actual.tolist(), predicted.tolist()
    idx = np.linspace(0, len(actual) - 1, n, dtype=int)
    return actual[idx].tolist(), predicted[idx].tolist()

scatter_data = {}

a, p = sample_arrays(lstm_actual, lstm_val_preds)
scatter_data["Multivariate LSTM (baseline)"] = {"actual": a, "predicted": p}

if HAS_XGB and xgb_model:
    a, p = sample_arrays(y_vl, xgb_val_preds)
    scatter_data["XGBoost + LSTM Ensemble"] = {"actual": a, "predicted": p}

# Feature importance
fi_data = {}
if HAS_XGB and xgb_model:
    importances = xgb_model.feature_importances_
    top_idx     = np.argsort(importances)[-15:][::-1]
    is_lstm     = [1 if FEATURE_COLS[i] in NEW_LSTM_COLS else 0 for i in top_idx]
    fi_data = {
        "features":     [FEATURE_COLS[i] for i in top_idx],
        "importances":  [round(float(importances[i]), 5) for i in top_idx],
        "is_lstm_meta": is_lstm,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  MATPLOTLIB PLOTS — saved as PNGs into models/plots/
#  These are served as static files by app.py and displayed in the frontend.
# ─────────────────────────────────────────────────────────────────────────────

MODEL_COLORS = {
    "Multivariate LSTM (baseline)":  "#34a853",
    "XGBoost + LSTM Ensemble":        "#5c6bc0",
}

PLOT_STYLE = {
    "figure.facecolor": "#ffffff",
    "axes.facecolor":   "#fafafa",
    "axes.grid":        True,
    "grid.color":       "#ececec",
    "grid.linewidth":   0.7,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "sans-serif",
    "axes.labelsize":   11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
}

plt.rcParams.update(PLOT_STYLE)

plot_paths = {}   # relative paths, served via /api/plots/


# ── Plot 1: RMSE Comparison bar chart ────────────────────────────────────────
print("\n[plots] Generating RMSE comparison bar chart...")
fig, ax = plt.subplots(figsize=(8, 4.5))
names  = list(results.keys())
rmses  = [results[n]["RMSE"] / 1e6 for n in names]
colors = [MODEL_COLORS.get(n, "#888") for n in names]
bars   = ax.bar(names, rmses, color=colors, width=0.5, zorder=3, edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, rmses):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f"€{val:.2f}M", ha="center", va="bottom", fontsize=9.5, fontweight="bold",
            color=bar.get_facecolor())
ax.set_ylabel("RMSE (€ Millions)")
ax.set_title("Validation RMSE by Model — Lower is Better")
ax.set_ylim(0, max(rmses) * 1.25)
short_names = [n.replace(" + LSTM Ensemble", "\n(Tuned)").replace(" (baseline)", "\n(Baseline)") for n in names]
ax.set_xticklabels(short_names, fontsize=10)
plt.tight_layout()
p = "plot_rmse_comparison.png"
fig.savefig(os.path.join(PLOTS_DIR, p), dpi=150, bbox_inches="tight")
plt.close(fig)
plot_paths["rmse_comparison"] = p
print(f"  ✅ Saved {p}")


# ── Plot 2: R² Comparison bar chart ──────────────────────────────────────────
print("[plots] Generating R² comparison bar chart...")
fig, ax = plt.subplots(figsize=(8, 4.5))
r2s = [results[n]["R2"] for n in names]
bars = ax.bar(names, r2s, color=colors, width=0.5, zorder=3, edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, r2s):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.4f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold",
            color=bar.get_facecolor())
ax.set_ylabel("R² Score")
ax.set_title("R² Score by Model — Higher is Better")
ax.set_ylim(0, 1.15)
ax.axhline(1.0, color="#ccc", linewidth=1, linestyle="--", zorder=2)
ax.set_xticklabels(short_names, fontsize=10)
plt.tight_layout()
p = "plot_r2_comparison.png"
fig.savefig(os.path.join(PLOTS_DIR, p), dpi=150, bbox_inches="tight")
plt.close(fig)
plot_paths["r2_comparison"] = p
print(f"  ✅ Saved {p}")


# ── Plot 3: Actual vs Predicted scatter — one subplot per model ───────────────
print("[plots] Generating actual vs predicted scatter plots...")
n_models = len(scatter_data)
fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 5))
if n_models == 1:
    axes = [axes]
for ax, (mname, sd) in zip(axes, scatter_data.items()):
    actual    = np.array(sd["actual"])    / 1e6
    predicted = np.array(sd["predicted"]) / 1e6
    color     = MODEL_COLORS.get(mname, "#888")
    ax.scatter(actual, predicted, color=color, alpha=0.45, s=18, edgecolors="none", zorder=3)
    lim = max(actual.max(), predicted.max()) * 1.05
    ax.plot([0, lim], [0, lim], color="#ccc", linewidth=1.5, linestyle="--", zorder=2)
    r2 = results[mname]["R2"] if mname in results else 0
    ax.set_title(f"{mname}\nR² = {r2:.4f}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Actual (€M)")
    ax.set_ylabel("Predicted (€M)")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
fig.suptitle("Actual vs Predicted Transfer Value — Validation Set", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
p = "plot_actual_vs_predicted.png"
fig.savefig(os.path.join(PLOTS_DIR, p), dpi=150, bbox_inches="tight")
plt.close(fig)
plot_paths["actual_vs_predicted"] = p
print(f"  ✅ Saved {p}")


# ── Plot 4: Feature Importance (XGBoost) ─────────────────────────────────────
if fi_data and fi_data.get("features"):
    print("[plots] Generating feature importance chart...")
    feats  = fi_data["features"]
    imps   = fi_data["importances"]
    is_lstm_m = fi_data["is_lstm_meta"]
    bar_colors = ["#fa7b17" if m else "#5c6bc0" for m in is_lstm_m]

    fig, ax = plt.subplots(figsize=(9, max(4, len(feats) * 0.42)))
    y_pos = range(len(feats))
    bars  = ax.barh(list(y_pos), imps, color=bar_colors, edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(feats, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title("XGBoost Feature Importance — Top Features")

    from matplotlib.patches import Patch
    legend_patches = [
        Patch(color="#fa7b17", label="LSTM meta-features"),
        Patch(color="#5c6bc0", label="Raw player features"),
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc="lower right")
    plt.tight_layout()
    p = "plot_feature_importance.png"
    fig.savefig(os.path.join(PLOTS_DIR, p), dpi=150, bbox_inches="tight")
    plt.close(fig)
    plot_paths["feature_importance"] = p
    print(f"  ✅ Saved {p}")


# ── Plot 5: MAE comparison ────────────────────────────────────────────────────
print("[plots] Generating MAE comparison chart...")
fig, ax = plt.subplots(figsize=(8, 4.5))
maes = [results[n]["MAE"] / 1e6 for n in names]
bars = ax.bar(names, maes, color=colors, width=0.5, zorder=3, edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, maes):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"€{val:.2f}M", ha="center", va="bottom", fontsize=9.5, fontweight="bold",
            color=bar.get_facecolor())
ax.set_ylabel("MAE (€ Millions)")
ax.set_title("Mean Absolute Error by Model — Lower is Better")
ax.set_ylim(0, max(maes) * 1.25)
ax.set_xticklabels(short_names, fontsize=10)
plt.tight_layout()
p = "plot_mae_comparison.png"
fig.savefig(os.path.join(PLOTS_DIR, p), dpi=150, bbox_inches="tight")
plt.close(fig)
plot_paths["mae_comparison"] = p
print(f"  ✅ Saved {p}")


# ── Plot 6: Combined dashboard (all 3 key metrics side by side) ───────────────
print("[plots] Generating combined metrics dashboard...")
fig = plt.figure(figsize=(16, 5))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

def _bar_ax(ax, vals, title, ylabel, fmt="€{:.2f}M", best_low=True):
    bar_list = ax.bar(names, vals, color=colors, width=0.45, zorder=3, edgecolor="white", linewidth=1.5)
    best_idx = (vals.index(min(vals)) if best_low else vals.index(max(vals)))
    for i, (bar, val) in enumerate(zip(bar_list, vals)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.02,
                fmt.format(val), ha="center", va="bottom", fontsize=8.5,
                fontweight="bold" if i == best_idx else "normal",
                color=bar.get_facecolor())
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(0, max(vals) * 1.3)
    ax.set_xticklabels([n.replace(" + LSTM Ensemble", "\n(Tuned)").replace(" (baseline)", "\n(Base)") for n in names], fontsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#ececec", linewidth=0.7)
    ax.set_axisbelow(True)

_bar_ax(fig.add_subplot(gs[0]), rmses,  "RMSE",           "€ Millions")
_bar_ax(fig.add_subplot(gs[1]), maes,   "MAE",            "€ Millions")
_bar_ax(fig.add_subplot(gs[2]), r2s,    "R² Score",       "Score",      fmt="{:.4f}", best_low=False)

fig.suptitle("Model Comparison Dashboard — Validation Set", fontsize=14, fontweight="bold", y=1.02)
p = "plot_dashboard.png"
fig.savefig(os.path.join(PLOTS_DIR, p), dpi=150, bbox_inches="tight")
plt.close(fig)
plot_paths["dashboard"] = p
print(f"  ✅ Saved {p}")

print(f"\n[plots] All {len(plot_paths)} plots saved to models/plots/")


# ─────────────────────────────────────────────────────────────────────────────
#  REPORT DATA
# ─────────────────────────────────────────────────────────────────────────────

report_data = {
    "generated_at": str(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")),
    "week":         "Evaluation & Hyperparameter Tuning",
    "n_players":    len(player_list),
    "n_train":      int(len(X_tr)),
    "n_val":        int(len(X_vl)),
    "n_features":   len(FEATURE_COLS),
    "new_column":   "lstm_predicted_market_value",
    "metrics":      {
        name: {
            "rmse_eur": round(m["RMSE"], 2),
            "mae_eur":  round(m["MAE"],  2),
            "r2":       round(m["R2"],   4),
        }
        for name, m in results.items()
    },
    "scatter":            scatter_data,
    "feature_importance": fi_data,
    "plots":              plot_paths,
    "tuning_results": {
        "XGBoost": {
            "method":      "GridSearchCV (5-fold, 54 combinations)",
            "best_params": xgb_best_params,
            "cv_rmse_eur": xgb_cv_rmse if HAS_XGB else None,
            "top_runs":    xgb_top_runs,
        },
    },
}

report_path = os.path.join(MODELS_DIR, "report_data.json")
with open(report_path, "w") as f:
    json.dump(report_data, f)

print(f"\n✅ New dataset  → player_transfer_value_ensemble.csv ({len(FEATURE_COLS)} features = {len(mv_features)} original + {len(NEW_LSTM_COLS)} LSTM columns)")
print(f"✅ Report data  → models/report_data.json")
print(f"✅ Plots        → models/plots/ ({len(plot_paths)} files)")