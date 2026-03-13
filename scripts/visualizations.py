# =============================================================================
# TransferIQ: Visualization Script
# =============================================================================
# This script generates three key visualizations for the TransferIQ project:
#   1. Transfer Value Trend over seasons for a specific player
#   2. Sentiment Score vs Market Value scatter plot with regression line
#   3. Predicted vs Actual transfer values scatter plot
#
# Libraries: matplotlib, seaborn, pandas
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set a clean visual style for all plots
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 11

# =============================================================================
# PLOT 1 — Transfer Value Trend (Line Chart)
# =============================================================================
# Shows how a player's market value changes across different seasons.
# Useful for tracking career progression, peak value, and decline.
#
# Expected columns in the DataFrame:
#   - player_name  : name of the player (string)
#   - season       : football season label, e.g. "2018/19" (string or int)
#   - market_value : transfer market value in euros (float)
# =============================================================================

def plot_transfer_value_trend(df, player_name):
    """
    Plot the transfer value trend for a single player over all available seasons.

    Parameters
    ----------
    df          : pandas DataFrame with columns [player_name, season, market_value]
    player_name : str — the exact name of the player to visualise
    """
    # Filter the dataframe to only keep rows for the chosen player
    player_df = df[df["player_name"] == player_name].copy()

    # Sort chronologically so the line goes left-to-right over time
    player_df = player_df.sort_values("season")

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # Draw the line and mark each data point
    ax.plot(
        player_df["season"],
        player_df["market_value"] / 1_000_000,   # convert to millions for readability
        marker="o",
        linewidth=2.5,
        color="#1f77b4",
        label=player_name
    )

    # Fill the area under the line for visual emphasis
    ax.fill_between(
        player_df["season"],
        player_df["market_value"] / 1_000_000,
        alpha=0.15,
        color="#1f77b4"
    )

    # Annotate each point with its value
    for _, row in player_df.iterrows():
        ax.annotate(
            f"€{row['market_value']/1e6:.1f}M",
            xy=(row["season"], row["market_value"] / 1_000_000),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            color="#333333"
        )

    # Labels and title
    ax.set_title(f"Transfer Value Trend — {player_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Season", fontsize=12)
    ax.set_ylabel("Market Value (€ Millions)", fontsize=12)
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("reports/figures/transfer_value_trend.png", bbox_inches="tight")
    plt.show()
    print(f"[Saved] reports/figures/transfer_value_trend.png")


# =============================================================================
# PLOT 2 — Sentiment Score vs Market Value (Scatter + Regression Line)
# =============================================================================
# Explores the relationship between a player's social media sentiment and
# their transfer market value.  A positive slope suggests that more positive
# sentiment correlates with higher valuations.
#
# Expected columns in the DataFrame:
#   - sentiment_score : numeric score, typically in range [-1, 1]
#   - market_value    : transfer market value in euros (float)
# =============================================================================

def plot_sentiment_vs_market_value(df):
    """
    Create a scatter plot with a regression line for sentiment vs market value.

    Parameters
    ----------
    df : pandas DataFrame with columns [sentiment_score, market_value]
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    # seaborn regplot draws both the scatter points and a linear regression line
    # with a shaded 95% confidence interval band automatically
    sns.regplot(
        data=df,
        x="sentiment_score",
        y="market_value",
        ax=ax,
        scatter_kws={"alpha": 0.5, "s": 60, "color": "#e67e22"},   # scatter style
        line_kws={"color": "#c0392b", "linewidth": 2},              # regression line style
        ci=95   # draw 95% confidence interval band
    )

    # Labels and title
    ax.set_title("Sentiment Score vs Transfer Market Value", fontsize=14, fontweight="bold")
    ax.set_xlabel("Sentiment Score  (−1 = Negative  →  +1 = Positive)", fontsize=11)
    ax.set_ylabel("Market Value (€)", fontsize=11)

    # Format y-axis ticks as millions
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda val, _: f"€{val/1e6:.0f}M")
    )

    plt.tight_layout()
    plt.savefig("reports/figures/sentiment_vs_market_value.png", bbox_inches="tight")
    plt.show()
    print("[Saved] reports/figures/sentiment_vs_market_value.png")


# =============================================================================
# PLOT 3 — Predicted vs Actual Values (Model Performance Scatter Plot)
# =============================================================================
# A standard diagnostic plot for regression models.
# Points close to the diagonal line = accurate predictions.
# Clusters above/below the line reveal systematic over- or under-prediction.
#
# Uses `y_test` (actual values) and `y_pred` (model predictions).
# =============================================================================

def plot_predicted_vs_actual(y_test, y_pred):
    """
    Scatter plot comparing model predictions against ground-truth values.
    A diagonal reference line represents perfect predictions (predicted == actual).

    Parameters
    ----------
    y_test : array-like — actual transfer market values (ground truth)
    y_pred : array-like — model predicted transfer market values
    """
    # Convert to numpy arrays for safety
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    fig, ax = plt.subplots(figsize=(8, 7))

    # Draw scatter points; colour-code by prediction error magnitude
    errors = np.abs(y_pred - y_test)
    scatter = ax.scatter(
        y_test / 1_000_000,
        y_pred / 1_000_000,
        c=errors / 1_000_000,      # colour by absolute error in millions
        cmap="RdYlGn_r",           # red = high error, green = low error
        alpha=0.7,
        s=60,
        edgecolors="k",
        linewidths=0.4
    )

    # Add a colour bar to show the error scale
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Absolute Error (€ Millions)", fontsize=10)

    # Draw the perfect-prediction diagonal reference line
    min_val = min(y_test.min(), y_pred.min()) / 1_000_000
    max_val = max(y_test.max(), y_pred.max()) / 1_000_000
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "b--",
        linewidth=2,
        label="Perfect Prediction (y = x)"
    )

    # Labels and title
    ax.set_title("Predicted vs Actual Transfer Market Values", fontsize=14, fontweight="bold")
    ax.set_xlabel("Actual Market Value (€ Millions)", fontsize=12)
    ax.set_ylabel("Predicted Market Value (€ Millions)", fontsize=12)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig("reports/figures/predicted_vs_actual.png", bbox_inches="tight")
    plt.show()
    print("[Saved] reports/figures/predicted_vs_actual.png")


# =============================================================================
# --- DEMO / QUICK TEST ---
# Replace the sample data below with your actual DataFrames and model outputs.
# =============================================================================

if __name__ == "__main__":

    # ── Sample data for Plot 1 ────────────────────────────────────────────────
    trend_data = pd.DataFrame({
        "player_name": ["L. Messi"] * 6,
        "season": ["2018/19", "2019/20", "2020/21", "2021/22", "2022/23", "2023/24"],
        "market_value": [120e6, 110e6, 80e6, 60e6, 50e6, 35e6]
    })
    plot_transfer_value_trend(trend_data, player_name="L. Messi")

    # ── Sample data for Plot 2 ────────────────────────────────────────────────
    np.random.seed(0)
    n = 200
    sentiment_data = pd.DataFrame({
        "sentiment_score": np.random.uniform(-1, 1, n),
        "market_value": np.random.uniform(1e6, 80e6, n)
    })
    # Add a mild positive correlation so the trend is visible
    sentiment_data["market_value"] += sentiment_data["sentiment_score"] * 15e6
    plot_sentiment_vs_market_value(sentiment_data)

    # ── Sample data for Plot 3 ────────────────────────────────────────────────
    y_test_sample = np.random.uniform(5e6, 80e6, 100)
    y_pred_sample = y_test_sample + np.random.normal(0, 5e6, 100)   # noisy predictions
    plot_predicted_vs_actual(y_test_sample, y_pred_sample)
