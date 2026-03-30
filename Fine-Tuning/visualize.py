"""
================================================================
visualize.py  —  Visualize all model metrics & tuning results
================================================================
Usage:
    cd "Fine-tuning/maybe"
    python visualize.py

Reads from ./tuned_models/ and saves a multi-page PDF +
individual PNGs to ./tuned_models/plots/
================================================================
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE, "tuned_models")
PLOTS_DIR  = os.path.join(MODELS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Theme ──────────────────────────────────────────────────────
plt.style.use("dark_background")

BG        = "#0e1117"
BG2       = "#1a1f2e"
ACCENT    = "#6c63ff"
GOLD      = "#ffd700"
GREEN     = "#00e676"
RED       = "#ff5252"
CYAN      = "#00bcd4"
ORANGE    = "#ff9800"
PINK      = "#e91e63"
LIME      = "#cddc39"
TEXT      = "#e0e0e0"
SUBTEXT   = "#9e9e9e"

MODEL_COLORS = {
    "Univariate LSTM"        : CYAN,
    "Multivariate LSTM"      : "#64b5f6",
    "Enc-Decoder LSTM (t+1)" : ORANGE,
    "Enc-Decoder LSTM (t+2)" : PINK,
    "XGBoost (tuned)"        : GREEN,
    "LightGBM (tuned)"       : LIME,
    "Ensemble (LSTM+XGB) ★"  : GOLD,
}

def euros(val, pos=None):
    """Format axis tick as €xM."""
    return f"€{val/1e6:.1f}M"

def pct(val, pos=None):
    return f"{val:.1f}%"

def styled_fig(w=16, h=9, title=""):
    fig = plt.figure(figsize=(w, h), facecolor=BG)
    if title:
        fig.suptitle(title, fontsize=18, fontweight="bold", color=TEXT,
                     y=0.98, fontfamily="monospace")
    return fig

def ax_style(ax, xlabel="", ylabel="", title=""):
    ax.set_facecolor(BG2)
    ax.tick_params(colors=SUBTEXT, labelsize=9)
    ax.spines[:].set_color("#2a2d3e")
    ax.grid(color="#2a2d3e", linewidth=0.6, linestyle="--")
    ax.set_xlabel(xlabel, color=SUBTEXT, fontsize=10)
    ax.set_ylabel(ylabel, color=SUBTEXT, fontsize=10)
    ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold", pad=10)

def save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"   💾  {name}")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  Football Player Valuation — Metrics Visualizer")
print("="*60)
print("📂  Loading artefacts...")

with open(os.path.join(MODELS_DIR, "model_metrics.json"))    as f: metrics    = json.load(f)
with open(os.path.join(MODELS_DIR, "feature_importance.json")) as f: fi_raw   = json.load(f)
with open(os.path.join(MODELS_DIR, "tuning_metadata.json"))  as f: meta      = json.load(f)

xgb_trials = pd.read_csv(os.path.join(MODELS_DIR, "tuning_results_xgb.csv"))
lgb_trials = pd.read_csv(os.path.join(MODELS_DIR, "tuning_results_lgb.csv"))

mdf = pd.DataFrame(metrics)
# Exclude the broken ensemble row from most charts (it skews scales badly)
mdf_clean = mdf[mdf["rmse"] < 5e6].copy()
fi        = pd.Series(fi_raw).sort_values(ascending=True)

models_clean = mdf_clean["model"].tolist()
colors_clean = [MODEL_COLORS.get(m, ACCENT) for m in models_clean]

print("✅  Data loaded.\n📊  Generating plots...\n")


# ════════════════════════════════════════════════════════════════
# FIGURE 1 — Model Comparison Dashboard
# ════════════════════════════════════════════════════════════════
fig = styled_fig(18, 10, "⚽  Football Player Valuation — Model Comparison Dashboard")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38,
                         left=0.06, right=0.97, top=0.92, bottom=0.08)

# — 1a: RMSE bar chart ─────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
bars = ax.barh(models_clean, mdf_clean["rmse"], color=colors_clean,
               edgecolor="none", height=0.6)
for bar, val in zip(bars, mdf_clean["rmse"]):
    ax.text(val + val*0.02, bar.get_y() + bar.get_height()/2,
            f"€{val/1e6:.2f}M", va="center", fontsize=8, color=TEXT)
ax.xaxis.set_major_formatter(FuncFormatter(euros))
ax_style(ax, xlabel="RMSE", title="RMSE  (lower = better)")
ax.invert_yaxis()

# — 1b: MAE bar chart ──────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
bars = ax.barh(models_clean, mdf_clean["mae"], color=colors_clean,
               edgecolor="none", height=0.6)
for bar, val in zip(bars, mdf_clean["mae"]):
    ax.text(val + val*0.02, bar.get_y() + bar.get_height()/2,
            f"€{val/1e6:.2f}M", va="center", fontsize=8, color=TEXT)
ax.xaxis.set_major_formatter(FuncFormatter(euros))
ax_style(ax, xlabel="MAE", title="MAE  (lower = better)")
ax.invert_yaxis()

# — 1c: R² bar chart ───────────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
bars = ax.barh(models_clean, mdf_clean["r2_pct"], color=colors_clean,
               edgecolor="none", height=0.6)
for bar, val in zip(bars, mdf_clean["r2_pct"]):
    ax.text(min(val + 0.005, 100.15), bar.get_y() + bar.get_height()/2,
            f"{val:.2f}%", va="center", fontsize=8, color=TEXT)
ax.set_xlim(99, 100.3)
ax_style(ax, xlabel="R²  (%)", title="R²  (higher = better)")
ax.invert_yaxis()

# — 1d: MAPE bar chart ─────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
bars = ax.barh(models_clean, mdf_clean["mape_pct"], color=colors_clean,
               edgecolor="none", height=0.6)
for bar, val in zip(bars, mdf_clean["mape_pct"]):
    ax.text(val + val*0.02, bar.get_y() + bar.get_height()/2,
            f"{val:.2f}%", va="center", fontsize=8, color=TEXT)
ax_style(ax, xlabel="MAPE (%)", title="MAPE  (lower = better)")
ax.invert_yaxis()

# — 1e: Directional Accuracy bar chart ────────────────────────
ax = fig.add_subplot(gs[1, 1])
bars = ax.barh(models_clean, mdf_clean["dir_acc_pct"], color=colors_clean,
               edgecolor="none", height=0.6)
for bar, val in zip(bars, mdf_clean["dir_acc_pct"]):
    ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}%", va="center", fontsize=8, color=TEXT)
ax.set_xlim(98, 101)
ax_style(ax, xlabel="Directional Accuracy (%)", title="Dir. Accuracy  (higher = better)")
ax.invert_yaxis()

# — 1f: Radar / Spider chart ───────────────────────────────────
ax_spider = fig.add_subplot(gs[1, 2], polar=True)
categories = ["R²", "Dir.Acc", "1-MAPE\n(norm)", "1-RMSE\n(norm)", "1-MAE\n(norm)"]
N = len(categories)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

for idx, row in mdf_clean.iterrows():
    vals = [
        (row["r2_pct"] - 99) / 1.0,             # R²: 99→100 → 0→1
        row["dir_acc_pct"] / 100.0,              # Dir Acc
        1 - row["mape_pct"] / mdf_clean["mape_pct"].max(),
        1 - row["rmse"] / mdf_clean["rmse"].max(),
        1 - row["mae"] / mdf_clean["mae"].max(),
    ]
    vals += vals[:1]
    c = MODEL_COLORS.get(row["model"], ACCENT)
    ax_spider.plot(angles, vals, color=c, linewidth=1.5, alpha=0.85)
    ax_spider.fill(angles, vals, color=c, alpha=0.07)

ax_spider.set_xticks(angles[:-1])
ax_spider.set_xticklabels(categories, color=TEXT, fontsize=8)
ax_spider.set_facecolor(BG2)
ax_spider.tick_params(colors=SUBTEXT)
ax_spider.grid(color="#2a2d3e", linewidth=0.6)
ax_spider.set_title("Radar  (all metrics)", color=TEXT, fontsize=11,
                     fontweight="bold", pad=20)

# legend
patches = [mpatches.Patch(color=MODEL_COLORS.get(m, ACCENT), label=m)
           for m in models_clean]
fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=8,
           framealpha=0.15, labelcolor=TEXT, facecolor=BG2,
           bbox_to_anchor=(0.5, 0.0))

save(fig, "01_model_comparison_dashboard.png")


# ════════════════════════════════════════════════════════════════
# FIGURE 2 — XGBoost Tuning Search
# ════════════════════════════════════════════════════════════════
fig = styled_fig(18, 10, "🔧  XGBoost Random Search — 20 Trials")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38,
                         left=0.07, right=0.97, top=0.90, bottom=0.08)

# RMSE over trials
ax = fig.add_subplot(gs[0, :2])
ax.plot(xgb_trials["trial"], xgb_trials["rmse"]/1e6, "o-",
        color=GREEN, linewidth=2, markersize=5, zorder=3)
best_running = xgb_trials["rmse"].cummin()
ax.plot(xgb_trials["trial"], best_running/1e6, "--",
        color=GOLD, linewidth=2, label="Best so far", zorder=4)
best_idx = xgb_trials["rmse"].idxmin()
ax.scatter(xgb_trials.loc[best_idx, "trial"],
           xgb_trials.loc[best_idx, "rmse"]/1e6,
           s=150, color=GOLD, zorder=5, label=f"Best trial (#{best_idx+1})")
ax.fill_between(xgb_trials["trial"], xgb_trials["rmse"]/1e6,
                best_running/1e6, color=GREEN, alpha=0.08)
ax_style(ax, xlabel="Trial", ylabel="RMSE (€M)", title="RMSE per Trial  vs  Running Best")
ax.yaxis.set_major_formatter(FuncFormatter(euros))
ax.legend(fontsize=9, framealpha=0.2, labelcolor=TEXT)
ax.set_xticks(xgb_trials["trial"])

# R² over trials
ax = fig.add_subplot(gs[0, 2])
ax.plot(xgb_trials["trial"], xgb_trials["r2"], "s-",
        color=CYAN, linewidth=2, markersize=5)
ax_style(ax, xlabel="Trial", ylabel="R²  (%)", title="R²  per Trial")
ax.yaxis.set_major_formatter(FuncFormatter(pct))
ax.set_xticks(xgb_trials["trial"][::4])

# Hyperparameter heatmap: learning_rate vs n_estimators → RMSE
ax = fig.add_subplot(gs[1, 0])
pivot = xgb_trials.pivot_table(values="rmse", index="learning_rate",
                                columns="n_estimators", aggfunc="min")
cmap = LinearSegmentedColormap.from_list("grd", [GREEN, GOLD, RED])
im = ax.imshow(pivot.values/1e6, cmap=cmap, aspect="auto")
ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, fontsize=8)
ax.set_yticks(range(len(pivot.index)));   ax.set_yticklabels(pivot.index, fontsize=8)
ax.set_xlabel("n_estimators", color=SUBTEXT, fontsize=9)
ax.set_ylabel("learning_rate", color=SUBTEXT, fontsize=9)
ax.set_title("RMSE Heatmap\nlr × n_estimators", color=TEXT, fontsize=10, fontweight="bold")
ax.set_facecolor(BG2)
fig.colorbar(im, ax=ax, format="%.1fM", shrink=0.8).ax.yaxis.set_tick_params(color=SUBTEXT)

# Scatter: subsample vs colsample_bytree coloured by RMSE
ax = fig.add_subplot(gs[1, 1])
sc = ax.scatter(xgb_trials["subsample"], xgb_trials["colsample_bytree"],
                c=xgb_trials["rmse"]/1e6, cmap=cmap, s=80, edgecolors="none", zorder=3)
ax_style(ax, xlabel="subsample", ylabel="colsample_bytree",
         title="RMSE by Subsample params")
fig.colorbar(sc, ax=ax, label="RMSE (€M)").ax.yaxis.set_tick_params(color=SUBTEXT)

# Best params table
ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
ax.set_facecolor(BG2)
bp = meta["xgboost"]["best_params"]
rows = [[k, str(v)] for k, v in bp.items()]
rows.insert(0, ["BEST RMSE", f"€{meta['xgboost']['best_rmse']/1e6:.3f}M"])
t = ax.table(cellText=rows, colLabels=["Parameter", "Value"],
             loc="center", cellLoc="center")
t.auto_set_font_size(False); t.set_fontsize(9)
for (r, c), cell in t.get_celld().items():
    cell.set_facecolor(BG2 if r > 0 else "#1e2a3a")
    cell.set_edgecolor("#2a2d3e")
    cell.set_text_props(color=TEXT if r > 0 else GOLD)
ax.set_title("Best Hyperparameters", color=TEXT, fontsize=11, fontweight="bold")

save(fig, "02_xgboost_tuning.png")


# ════════════════════════════════════════════════════════════════
# FIGURE 3 — LightGBM Tuning Search
# ════════════════════════════════════════════════════════════════
fig = styled_fig(18, 10, "🔧  LightGBM Random Search — 20 Trials")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38,
                         left=0.07, right=0.97, top=0.90, bottom=0.08)

# RMSE over trials
ax = fig.add_subplot(gs[0, :2])
ax.plot(lgb_trials["trial"], lgb_trials["rmse"]/1e6, "o-",
        color=LIME, linewidth=2, markersize=5, zorder=3)
best_running_l = lgb_trials["rmse"].cummin()
ax.plot(lgb_trials["trial"], best_running_l/1e6, "--",
        color=GOLD, linewidth=2, label="Best so far")
best_idx_l = lgb_trials["rmse"].idxmin()
ax.scatter(lgb_trials.loc[best_idx_l, "trial"],
           lgb_trials.loc[best_idx_l, "rmse"]/1e6,
           s=150, color=GOLD, zorder=5, label=f"Best trial (#{best_idx_l+1})")
ax.fill_between(lgb_trials["trial"], lgb_trials["rmse"]/1e6,
                best_running_l/1e6, color=LIME, alpha=0.08)
ax_style(ax, xlabel="Trial", ylabel="RMSE (€M)", title="RMSE per Trial  vs  Running Best")
ax.yaxis.set_major_formatter(FuncFormatter(euros))
ax.legend(fontsize=9, framealpha=0.2, labelcolor=TEXT)
ax.set_xticks(lgb_trials["trial"])

# R² over trials
ax = fig.add_subplot(gs[0, 2])
ax.plot(lgb_trials["trial"], lgb_trials["r2"], "s-",
        color=ORANGE, linewidth=2, markersize=5)
ax_style(ax, xlabel="Trial", ylabel="R²  (%)", title="R²  per Trial")
ax.yaxis.set_major_formatter(FuncFormatter(pct))
ax.set_xticks(lgb_trials["trial"][::4])

# Heatmap: num_leaves vs learning_rate → RMSE
ax = fig.add_subplot(gs[1, 0])
pivot_l = lgb_trials.pivot_table(values="rmse", index="num_leaves",
                                  columns="learning_rate", aggfunc="min")
im2 = ax.imshow(pivot_l.values/1e6, cmap=cmap, aspect="auto")
ax.set_xticks(range(len(pivot_l.columns))); ax.set_xticklabels(pivot_l.columns, fontsize=8)
ax.set_yticks(range(len(pivot_l.index)));   ax.set_yticklabels(pivot_l.index, fontsize=8)
ax.set_xlabel("learning_rate", color=SUBTEXT, fontsize=9)
ax.set_ylabel("num_leaves", color=SUBTEXT, fontsize=9)
ax.set_title("RMSE Heatmap\nnum_leaves × lr", color=TEXT, fontsize=10, fontweight="bold")
ax.set_facecolor(BG2)
fig.colorbar(im2, ax=ax, format="%.1fM", shrink=0.8).ax.yaxis.set_tick_params(color=SUBTEXT)

# Scatter: subsample vs colsample_bytree
ax = fig.add_subplot(gs[1, 1])
sc2 = ax.scatter(lgb_trials["subsample"], lgb_trials["colsample_bytree"],
                 c=lgb_trials["rmse"]/1e6, cmap=cmap, s=80, edgecolors="none", zorder=3)
ax_style(ax, xlabel="subsample", ylabel="colsample_bytree",
         title="RMSE by Subsample params")
fig.colorbar(sc2, ax=ax, label="RMSE (€M)").ax.yaxis.set_tick_params(color=SUBTEXT)

# Best params table
ax = fig.add_subplot(gs[1, 2])
ax.axis("off"); ax.set_facecolor(BG2)
bp_l = meta["lightgbm"]["best_params"]
rows_l = [[k, str(v)] for k, v in bp_l.items()]
rows_l.insert(0, ["BEST RMSE", f"€{meta['lightgbm']['best_rmse']/1e6:.3f}M"])
t2 = ax.table(cellText=rows_l, colLabels=["Parameter", "Value"],
              loc="center", cellLoc="center")
t2.auto_set_font_size(False); t2.set_fontsize(9)
for (r, c), cell in t2.get_celld().items():
    cell.set_facecolor(BG2 if r > 0 else "#1e2a3a")
    cell.set_edgecolor("#2a2d3e")
    cell.set_text_props(color=TEXT if r > 0 else LIME)
ax.set_title("Best Hyperparameters", color=TEXT, fontsize=11, fontweight="bold")

save(fig, "03_lightgbm_tuning.png")


# ════════════════════════════════════════════════════════════════
# FIGURE 4 — Feature Importance
# ════════════════════════════════════════════════════════════════
fig = styled_fig(16, 8, "📊  XGBoost Feature Importance  (Top 25 Features)")
ax  = fig.add_subplot(111)

# Color features by category
def feat_color(name):
    if "market" in name or "log_market" in name:           return GOLD
    if "prev_value" in name or "value_trend" in name \
            or "pct_change" in name:                        return GREEN
    if "age" in name:                                       return CYAN
    if any(x in name for x in ["tweet","social","vader",
                                "sentiment","tb_","positive",
                                "negative","neutral"]):      return PINK
    if any(x in name for x in ["injury","availab"]):        return RED
    return ACCENT

colors_fi = [feat_color(n) for n in fi.index]
bars = ax.barh(fi.index, fi.values * 100, color=colors_fi, edgecolor="none", height=0.7)
for bar, val in zip(bars, fi.values):
    ax.text(val*100 + 0.05, bar.get_y() + bar.get_height()/2,
            f"{val*100:.2f}%", va="center", fontsize=8, color=TEXT)

ax_style(ax, xlabel="Importance (%)", title="")
ax.set_facecolor(BG2)

legend_items = [
    mpatches.Patch(color=GOLD,  label="Market value features"),
    mpatches.Patch(color=GREEN, label="Lag / trend features"),
    mpatches.Patch(color=CYAN,  label="Age features"),
    mpatches.Patch(color=PINK,  label="Sentiment features"),
    mpatches.Patch(color=RED,   label="Injury features"),
    mpatches.Patch(color=ACCENT,label="Performance features"),
]
ax.legend(handles=legend_items, loc="lower right", fontsize=9,
          framealpha=0.15, labelcolor=TEXT, facecolor=BG2)

fig.tight_layout(rect=[0, 0, 1, 0.95])
save(fig, "04_feature_importance.png")


# ════════════════════════════════════════════════════════════════
# FIGURE 5 — XGB vs LGB trial RMSE comparison
# ════════════════════════════════════════════════════════════════
fig = styled_fig(16, 7, "⚖️  XGBoost vs LightGBM — Trial-by-Trial RMSE Comparison")
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35,
                         left=0.07, right=0.97, top=0.88, bottom=0.10)

# side-by-side line plot (excluding extreme outlier trials >5M)
ax = fig.add_subplot(gs[0])
safe_xgb = xgb_trials["rmse"].clip(upper=3e6)   # cap for visual clarity
safe_lgb = lgb_trials["rmse"].clip(upper=3e6)
ax.plot(xgb_trials["trial"], safe_xgb/1e6, "o-", color=GREEN,
        linewidth=2, markersize=5, label="XGBoost")
ax.plot(lgb_trials["trial"], safe_lgb/1e6, "s-", color=LIME,
        linewidth=2, markersize=5, label="LightGBM")
ax.axhline(meta["xgboost"]["best_rmse"]/1e6, color=GREEN, linestyle=":",
           linewidth=1.5, label=f'XGB best €{meta["xgboost"]["best_rmse"]/1e6:.2f}M')
ax.axhline(meta["lightgbm"]["best_rmse"]/1e6, color=LIME, linestyle=":",
           linewidth=1.5, label=f'LGB best €{meta["lightgbm"]["best_rmse"]/1e6:.2f}M')
ax_style(ax, xlabel="Trial", ylabel="RMSE (€M)",
         title="RMSE per Trial  (capped at €3M for clarity)")
ax.yaxis.set_major_formatter(FuncFormatter(euros))
ax.legend(fontsize=9, framealpha=0.2, labelcolor=TEXT)
ax.set_xticks(xgb_trials["trial"])

# Box plot of RMSE distributions
ax = fig.add_subplot(gs[1])
data_box = [xgb_trials["rmse"][xgb_trials["rmse"] < 3e6]/1e6,
            lgb_trials["rmse"][lgb_trials["rmse"] < 3e6]/1e6]
bp = ax.boxplot(data_box, patch_artist=True, widths=0.45,
                medianprops=dict(color=GOLD, linewidth=2),
                whiskerprops=dict(color=SUBTEXT),
                capprops=dict(color=SUBTEXT),
                flierprops=dict(marker="o", color=RED, alpha=0.5, markersize=4))
for patch, color in zip(bp["boxes"], [GREEN, LIME]):
    patch.set_facecolor(color); patch.set_alpha(0.5)
ax.set_xticks([1, 2]); ax.set_xticklabels(["XGBoost", "LightGBM"], color=TEXT, fontsize=10)
ax_style(ax, ylabel="RMSE (€M)", title="RMSE Distribution\n(trials ≤ €3M)")
ax.yaxis.set_major_formatter(FuncFormatter(euros))

save(fig, "05_xgb_vs_lgbm_tuning.png")


# ════════════════════════════════════════════════════════════════
# FIGURE 6 — Full Metrics Summary Table
# ════════════════════════════════════════════════════════════════
fig = styled_fig(14, 5, "📋  Full Evaluation Summary — All Models")
ax  = fig.add_subplot(111)
ax.axis("off"); ax.set_facecolor(BG)

col_labels = ["Model", "RMSE", "MAE", "R²%", "MAPE%", "Dir.Acc%"]
table_data = []
for row in metrics:
    table_data.append([
        row["model"],
        f"€{row['rmse']/1e6:.2f}M",
        f"€{row['mae']/1e6:.2f}M",
        f"{row['r2_pct']:.2f}%",
        f"{row['mape_pct']:.2f}%",
        f"{row['dir_acc_pct']:.1f}%",
    ])

t = ax.table(cellText=table_data, colLabels=col_labels,
             loc="center", cellLoc="center")
t.auto_set_font_size(False); t.set_fontsize(10)
t.scale(1, 2.0)

for (r, c), cell in t.get_celld().items():
    if r == 0:
        cell.set_facecolor("#1e2a3a"); cell.set_text_props(color=GOLD, fontweight="bold")
    elif table_data[r-1][0] in MODEL_COLORS:
        m = table_data[r-1][0]
        cell.set_facecolor(BG2)
        if c == 0:
            cell.set_text_props(color=MODEL_COLORS.get(m, TEXT), fontweight="bold")
        else:
            cell.set_text_props(color=TEXT)
    else:
        cell.set_facecolor(BG2); cell.set_text_props(color=TEXT)
    cell.set_edgecolor("#2a2d3e")

fig.tight_layout(rect=[0, 0, 1, 0.93])
save(fig, "06_metrics_summary_table.png")


# ════════════════════════════════════════════════════════════════
# DONE
# ════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  ✅  All plots saved to:  ./tuned_models/plots/")
print(f"{'='*60}\n")
print("  📁  Files generated:")
for f in sorted(os.listdir(PLOTS_DIR)):
    size = os.path.getsize(os.path.join(PLOTS_DIR, f)) // 1024
    print(f"       {f:<50}  {size} KB")
print()
