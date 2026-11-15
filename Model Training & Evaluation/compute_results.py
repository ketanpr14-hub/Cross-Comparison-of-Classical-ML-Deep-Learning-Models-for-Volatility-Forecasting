"""
Plot initial QLIKE results for the naive and HAR (OLS) models.

Reads:  results/forecasts/<asset_class>_<symbol>.parquet  (from baseline_models.py)
Writes: results/figures/qlike_by_asset_class.html
        results/figures/qlike_ratio.html

Figure 1: QLIKE per symbol, naive vs HAR, one panel per asset class.
Figure 2: HAR's QLIKE as a ratio to naive (below 1 = HAR is better).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error

RESULTS_DIR = Path("results/forecasts")
FIG_DIR = Path("results/figures")
ASSET_CLASSES = ["fx", "futures", "stocks"]

# Fixed colors per model, so every chart uses the same one for each
MODEL_STYLE = {
    "naive": {"label": "Naive", "color": "#2a78d6"},
    "har":   {"label": "HAR (OLS)", "color": "#eb6834"},
}
INK, INK_MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"


def qlike(actual, forecast):
    """QLIKE on variances; inputs are log vols."""
    ratio = np.exp(2 * actual) / np.exp(2 * forecast)
    return np.mean(ratio - np.log(ratio) - 1)


def load_summary() -> pd.DataFrame:
    forecasts = pd.concat(
        [pd.read_parquet(p) for p in sorted(RESULTS_DIR.glob("*.parquet"))],
        ignore_index=True,
    )
    summary = (
        forecasts.groupby(["asset_class", "symbol", "model"])
        .apply(lambda g: qlike(g["actual"], g["forecast"]), include_groups=False)
        .rename("qlike")
        .reset_index()
    )
    naive = summary[summary["model"] == "naive"].set_index(["asset_class", "symbol"])["qlike"]
    summary["ratio"] = summary["qlike"] / summary.set_index(["asset_class", "symbol"]).index.map(naive)
    return summary


def style(fig: go.Figure, title: str, height: int) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=INK), x=0, xanchor="left"),
        template="plotly_white",
        height=height,
        font=dict(family="Inter, Arial, sans-serif", size=12, color=INK),
        legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="left", x=0),
        margin=dict(l=70, r=30, t=120, b=50),
        bargap=0.35,
        bargroupgap=0.08,
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickfont=dict(color=INK_MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont=dict(color=INK_MUTED))
    for ann in fig.layout.annotations:  # subplot titles
        ann.font = dict(size=13, color=INK)
    return fig


def plot_qlike(summary: pd.DataFrame) -> go.Figure:
    """Raw QLIKE per symbol, one panel per asset class. Each panel has its own
    y-axis because QLIKE levels differ across asset classes."""
    classes = [c for c in ASSET_CLASSES if c in summary["asset_class"].unique()]
    fig = make_subplots(rows=1, cols=len(classes), subplot_titles=[c.upper() for c in classes],
                        horizontal_spacing=0.08)

    for i, asset_class in enumerate(classes, start=1):
        sub = summary[summary["asset_class"] == asset_class]
        for model, s in MODEL_STYLE.items():
            m = sub[sub["model"] == model]
            fig.add_bar(
                x=m["symbol"], y=m["qlike"], name=s["label"], marker_color=s["color"],
                marker_line_width=0, legendgroup=model, showlegend=(i == 1), row=1, col=i,
                hovertemplate=f"<b>%{{x}}</b><br>{s['label']}<br>QLIKE %{{y:.4f}}<extra></extra>",
            )
    fig.update_yaxes(title_text="QLIKE (lower is better)", row=1, col=1)
    return style(fig, "QLIKE by symbol: naive vs HAR", height=460)


def plot_ratio(summary: pd.DataFrame) -> go.Figure:
    """HAR QLIKE / naive QLIKE per symbol. Below 1 means HAR beats naive."""
    har = summary[summary["model"] == "har"].copy()
    har["order"] = har["asset_class"].map({c: i for i, c in enumerate(ASSET_CLASSES)})
    har = har.sort_values(["order", "ratio"])
    labels = har["asset_class"].str.upper() + " · " + har["symbol"]

    fig = go.Figure()
    fig.add_vline(x=1, line_color=INK_MUTED, line_width=1, line_dash="dash")
    fig.add_scatter(
        x=har["ratio"], y=labels, mode="markers",
        marker=dict(size=11, color=MODEL_STYLE["har"]["color"], line=dict(color="white", width=2)),
        name="HAR (OLS)", showlegend=False,
        hovertemplate="<b>%{y}</b><br>HAR / naive QLIKE: %{x:.3f}<extra></extra>",
    )
    fig.add_annotation(x=1, y=1, yref="paper", text="Naive = 1", showarrow=False,
                       yanchor="bottom", font=dict(color=INK_MUTED, size=11))
    fig = style(fig, "HAR QLIKE relative to naive (below 1 = HAR is better)",
                height=max(320, 40 * len(har) + 140))
    fig.update_xaxes(showgrid=True, gridcolor=GRID, title_text="QLIKE ratio to naive")
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary = load_summary()
    print(summary.pivot_table(index=["asset_class", "symbol"], columns="model", values="qlike").round(4))

    fig1 = plot_qlike(summary)
    fig2 = plot_ratio(summary)
    fig1.write_html(FIG_DIR / "qlike_by_asset_class.html")
    fig2.write_html(FIG_DIR / "qlike_ratio.html")
    fig1.show()
    fig2.show()


if __name__ == "__main__":
    main()