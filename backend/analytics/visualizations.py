"""
analytics/visualizations.py
-----------------------------
Matplotlib visualizations rendered to base64-encoded PNG strings.
No files written to disk — each chart is returned as a data URI
ready for an HTML <img src="data:image/png;base64,..."> tag.

Charts
------
1. bar_completion_by_time_of_day(df)
       Bar chart: completion rate for each time-of-day slot.

2. heatmap_energy_vs_completion(df)
       Heatmap: average completion rate for every
       energy_level × task_type combination.

Public API
----------
  generate_all_charts(df) → dict { chart_name: base64_string }
"""

from __future__ import annotations

import base64
import io

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Styling constants — Blue Haze palette (matches frontend light theme)
# ---------------------------------------------------------------------------

_FONT_FAMILY  = "DejaVu Sans"
_BG_COLOR     = "#fdf6ee"          # --surface-raised: warm cream
_PANEL_COLOR  = "#ffffff"          # --surface: white
_GRID_COLOR   = "#e8ddd0"          # warm cream border
_TEXT_COLOR   = "#102a6b"          # --text-primary: Silent Navy
_TEXT_MUTED   = "#7a8fb5"          # --text-muted
_ACCENT       = "#cea273"          # --accent: Sandy Amber
_STEEL        = "#5990c0"          # --steel: Steel Blue
_NAVY         = "#015185"          # --accent2: Blue Current

# Bar chart: three shades from Sandy Amber → Steel Blue
_BAR_PALETTE  = ["#102a6b", "#5990c0", "#a8c4e0"]

# Heatmap: cream → navy (matches the palette direction)
_HEATMAP_CMAP = "YlGnBu"

plt.rcParams.update({
    "font.family":       _FONT_FAMILY,
    "axes.facecolor":    _BG_COLOR,
    "figure.facecolor":  _PANEL_COLOR,
    "axes.edgecolor":    _GRID_COLOR,
    "axes.grid":         True,
    "grid.color":        _GRID_COLOR,
    "grid.linewidth":    0.8,
    "text.color":        _TEXT_COLOR,
    "axes.labelcolor":   _TEXT_COLOR,
    "xtick.color":       _TEXT_COLOR,
    "ytick.color":       _TEXT_COLOR,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate(df: pd.DataFrame) -> None:
    required = {"completed", "time_of_day", "energy_level", "task_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def _to_base64(fig: plt.Figure) -> str:
    """Render a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


# ---------------------------------------------------------------------------
# Chart 1: Bar chart — completion rate by time_of_day
# ---------------------------------------------------------------------------

def bar_completion_by_time_of_day(df: pd.DataFrame) -> str:
    """
    Bar chart showing completion rate for each time-of-day slot,
    ordered morning → afternoon → night.

    Returns
    -------
    str : base64-encoded PNG
    """
    _validate(df)

    # Compute rates
    rates = (
        df.groupby("time_of_day")["completed"]
        .mean()
        .reindex(["morning", "afternoon", "night"])  # fixed display order
        .dropna()
        .mul(100)
        .round(1)
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))

    bars = ax.bar(
        rates.index,
        rates.values,
        color=_BAR_PALETTE[: len(rates)],
        width=0.5,
        zorder=3,
    )

    # Value labels on top of each bar
    for bar, val in zip(bars, rates.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=_TEXT_COLOR,
        )

    ax.set_title("Completion Rate by Time of Day", fontsize=13,
                 fontweight="bold", pad=14, color=_TEXT_COLOR)
    ax.set_xlabel("Time of Day", fontsize=10, labelpad=8)
    ax.set_ylabel("Completion Rate (%)", fontsize=10, labelpad=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_ylim(0, min(rates.max() * 1.25, 100))
    ax.tick_params(axis="both", labelsize=9)
    fig.tight_layout()
    return _to_base64(fig)


# ---------------------------------------------------------------------------
# Chart 2: Heatmap — energy_level × task_type → completion rate
# ---------------------------------------------------------------------------

def heatmap_energy_vs_completion(df: pd.DataFrame) -> str:
    """
    Heatmap of average completion rate for every energy_level (1–5) ×
    task_type combination. Darker cells = higher completion rate.

    Returns
    -------
    str : base64-encoded PNG
    """
    _validate(df)

    # Pivot: rows = energy_level (5 → 1), cols = task_type
    pivot = (
        df.groupby(["energy_level", "task_type"])["completed"]
        .mean()
        .unstack(fill_value=np.nan)
        .sort_index(ascending=False)   # energy 5 at top
    )

    # Desired column order
    col_order = [c for c in ["deep_work", "creative", "admin"] if c in pivot.columns]
    pivot = pivot[col_order]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    im = ax.imshow(
        pivot.values,
        cmap=_HEATMAP_CMAP,
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    # Axis labels
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(
        [c.replace("_", " ").title() for c in pivot.columns],
        fontsize=10,
    )
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(
        [f"Energy {int(e)}" for e in pivot.index],
        fontsize=10,
    )

    # Annotate each cell with the completion rate
    for row_idx in range(pivot.shape[0]):
        for col_idx in range(pivot.shape[1]):
            val = pivot.values[row_idx, col_idx]
            if not np.isnan(val):
                # Use white text on dark cells, navy on light cells
                text_color = "#ffffff" if val > 0.50 else _TEXT_COLOR
                ax.text(
                    col_idx, row_idx,
                    f"{val * 100:.1f}%",
                    ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color=text_color,
                )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Completion Rate", fontsize=9)
    cbar.ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x * 100:.0f}%")
    )

    ax.set_title("Completion Rate: Energy Level × Task Type",
                 fontsize=13, fontweight="bold", pad=14, color=_TEXT_COLOR)
    ax.set_xlabel("Task Type", fontsize=10, labelpad=8)
    ax.set_ylabel("Energy Level", fontsize=10, labelpad=8)
    ax.grid(False)

    fig.tight_layout()
    return _to_base64(fig)


# ---------------------------------------------------------------------------
# Generate all charts
# ---------------------------------------------------------------------------

def generate_all_charts(df: pd.DataFrame) -> dict[str, str]:
    """
    Render all charts and return a dict of base64-encoded PNG strings.

    Returns
    -------
    dict
        {
            "bar_completion_by_time_of_day":  "<base64>",
            "heatmap_energy_vs_completion":   "<base64>",
        }
    """
    _validate(df)

    return {
        "bar_completion_by_time_of_day": bar_completion_by_time_of_day(df),
        "heatmap_energy_vs_completion":  heatmap_energy_vs_completion(df),
    }
