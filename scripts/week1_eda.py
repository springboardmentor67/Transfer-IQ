from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _load_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _dataset_section(name: str, df: pd.DataFrame, figures_dir: Path) -> str:
    lines: List[str] = []
    lines.append(f"## {name}\n")
    lines.append(f"Rows: {len(df):,}  |  Columns: {df.shape[1]:,}\n")

    lines.append("### Columns & Types\n")
    dtypes = pd.DataFrame({"column": df.columns, "dtype": [str(t) for t in df.dtypes]})
    lines.append(dtypes.to_markdown(index=False))
    lines.append("\n")

    lines.append("### Missing Values\n")
    miss = (df.isna().mean() * 100).sort_values(ascending=False).reset_index()
    miss.columns = ["column", "missing_%"]
    lines.append(miss.to_markdown(index=False))
    lines.append("\n")

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if num_cols:
        lines.append("### Numeric Summary\n")
        lines.append(df[num_cols].describe().transpose().to_markdown())
        lines.append("\n")

        plot_cols = num_cols[:8]
        for col in plot_cols:
            fig_path = figures_dir / f"{name.lower().replace(' ', '_')}_{col}_hist.png"
            plt.figure(figsize=(7, 4))
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f"{name}: {col}")
            plt.tight_layout()
            plt.savefig(fig_path, dpi=150)
            plt.close()
            lines.append(f"- Figure: {fig_path.as_posix()}\n")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 1: initial EDA (distributions + missing data) and exploration report.")
    parser.add_argument("--processed", default="data/processed", help="Processed data directory")
    parser.add_argument("--raw", default="data/raw", help="Raw data directory")
    parser.add_argument("--report", default="reports/week1_exploration_report.md")
    parser.add_argument("--figures", default="reports/figures")
    args = parser.parse_args()

    processed_dir = Path(args.processed)
    raw_dir = Path(args.raw)
    report_path = Path(args.report)
    figures_dir = Path(args.figures)
    figures_dir.mkdir(parents=True, exist_ok=True)

    datasets: List[Tuple[str, Optional[pd.DataFrame], str]] = [
        (
            "Player Data",
            _load_if_exists(Path("data/player_data.csv")),
            "data/player_data.csv",
        ),
    ]

    lines: List[str] = []
    lines.append("# Week 1 — Data Exploration Report\n")
    lines.append(f"Generated at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n")

    lines.append("## Inventory\n")
    inv_rows = []
    for name, df, path in datasets:
        inv_rows.append(
            {
                "dataset": name,
                "path": path,
                "status": "OK" if df is not None else "MISSING",
                "rows": int(len(df)) if df is not None else None,
                "columns": int(df.shape[1]) if df is not None else None,
            }
        )
    lines.append(pd.DataFrame(inv_rows).to_markdown(index=False))
    lines.append("\n")

    for name, df, _ in datasets:
        if df is None:
            lines.append(f"## {name}\n")
            lines.append("Dataset not found yet. Run the corresponding Week 1 collection script.\n")
            continue
        lines.append(_dataset_section(name, df, figures_dir))

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
