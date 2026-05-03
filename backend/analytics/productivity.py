"""
analytics/productivity.py
--------------------------
Functions for analysing the productivity dataset.

All public functions accept a pandas DataFrame that matches the schema of
data/productivity.csv and return plain Python dicts so results are easy to
serialise, log, or pass to downstream code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_ENERGY_MIN, _ENERGY_MAX = 1, 5
_FOCUS_MIN, _FOCUS_MAX = 1, 5
_DISTRACTION_MIN, _DISTRACTION_MAX = 0, 15

# Weights used in productivity_score (must sum to 1.0)
_W_COMPLETION = 0.40
_W_ENERGY = 0.25
_W_FOCUS = 0.25
_W_DISTRACTION = 0.10  # penalty component


def _normalise(series: pd.Series, lo: float, hi: float) -> pd.Series:
    """Min-max normalise a series to [0, 1] using known domain bounds."""
    return (series - lo) / (hi - lo)


def _validate(df: pd.DataFrame) -> None:
    required = {
        "completed", "time_of_day", "task_type",
        "energy_level", "focus_level", "distraction_count",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


# ---------------------------------------------------------------------------
# Public analytics functions
# ---------------------------------------------------------------------------

def completion_rate_by_time_of_day(df: pd.DataFrame) -> dict[str, float]:
    """
    Completion rate (0-1) for each time-of-day bucket.

    Returns
    -------
    dict
        Keys are time-of-day labels; values are completion rates rounded to 4 dp.
        Example::

            {
                "morning":   0.4773,
                "afternoon": 0.2874,
                "night":     0.1695,
            }
    """
    _validate(df)
    rates = (
        df.groupby("time_of_day")["completed"]
        .mean()
        .round(4)
        .to_dict()
    )
    return rates


def completion_rate_by_task_type(df: pd.DataFrame) -> dict[str, float]:
    """
    Completion rate (0-1) for each task type.

    Returns
    -------
    dict
        Keys are task-type labels; values are completion rates rounded to 4 dp.
        Example::

            {
                "deep_work": 0.4878,
                "admin":     0.2400,
                "creative":  0.2333,
            }
    """
    _validate(df)
    rates = (
        df.groupby("task_type")["completed"]
        .mean()
        .round(4)
        .to_dict()
    )
    return rates


def correlations(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """
    Pearson and point-biserial correlations between key numeric features and
    the binary ``completed`` column.

    Because ``completed`` is binary (0/1), the Pearson correlation is
    mathematically equivalent to the point-biserial correlation, so both are
    reported from the same calculation for transparency.

    Also computes a full correlation matrix across all numeric features using
    numpy.corrcoef, giving a simultaneous view of how every variable relates
    to every other variable.

    Returns
    -------
    dict
        Structure::

            {
                "energy_level_vs_completed": {
                    "pearson": 0.xxxx,
                    "point_biserial": 0.xxxx,
                    "interpretation": "..."
                },
                "distraction_count_vs_completed": { ... },
                "correlation_matrix": {
                    "variables": [...],
                    "matrix":    [[...], ...]   # row/col order matches variables
                }
            }
    """
    _validate(df)

    def _corr_entry(col: str) -> dict[str, float | str]:
        r = round(df[col].corr(df["completed"]), 4)
        if abs(r) < 0.10:
            interp = "negligible"
        elif abs(r) < 0.30:
            interp = "weak"
        elif abs(r) < 0.50:
            interp = "moderate"
        else:
            interp = "strong"
        direction = "positive" if r >= 0 else "negative"
        return {
            "pearson": r,
            "point_biserial": r,
            "interpretation": f"{direction} {interp}",
        }

    # Full correlation matrix via numpy.corrcoef
    corr_cols = ["energy_level", "focus_level", "distraction_count", "duration", "completed"]
    available = [c for c in corr_cols if c in df.columns]
    matrix_data = np.corrcoef(df[available].dropna().to_numpy().T)
    corr_matrix = {
        "variables": available,
        "matrix": [
            [round(float(v), 4) for v in row]
            for row in matrix_data
        ],
    }

    return {
        "energy_level_vs_completed":      _corr_entry("energy_level"),
        "distraction_count_vs_completed": _corr_entry("distraction_count"),
        "correlation_matrix":             corr_matrix,
    }


def productivity_score(df: pd.DataFrame) -> pd.Series:
    """
    Compute a per-row productivity score in **[0, 1]**.

    Formula
    -------
    Each component is first normalised to [0, 1] using its known domain range,
    then combined as a weighted sum::

        score = 0.40 * completed
              + 0.25 * norm(energy_level,      1, 5)
              + 0.25 * norm(focus_level,        1, 5)
              + 0.10 * (1 - norm(distraction_count, 0, 15))   # penalty

    Weights reflect that task completion is the primary outcome, while energy
    and focus are equal secondary drivers, and distraction is a minor penalty.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``completed``, ``energy_level``, ``focus_level``,
        ``distraction_count``.

    Returns
    -------
    pd.Series
        Float scores in [0, 1], rounded to 4 decimal places, named
        ``"productivity_score"``.
    """
    _validate(df)

    norm_energy = _normalise(df["energy_level"], _ENERGY_MIN, _ENERGY_MAX)
    norm_focus = _normalise(df["focus_level"], _FOCUS_MIN, _FOCUS_MAX)
    norm_distraction = _normalise(
        df["distraction_count"], _DISTRACTION_MIN, _DISTRACTION_MAX
    )

    score = (
        _W_COMPLETION * df["completed"]
        + _W_ENERGY * norm_energy
        + _W_FOCUS * norm_focus
        + _W_DISTRACTION * (1 - norm_distraction)  # invert: fewer = better
    )

    return score.round(4).rename("productivity_score")


def full_report(df: pd.DataFrame) -> dict:
    """
    Run all analytics in one call and return a single consolidated dictionary.

    Returns
    -------
    dict
        Structure::

            {
                "completion_rate_by_time_of_day": { ... },
                "completion_rate_by_task_type":   { ... },
                "correlations":                   { ... },
                "productivity_score": {
                    "mean":   float,
                    "median": float,
                    "std":    float,
                    "min":    float,
                    "max":    float,
                    "by_task_type":   { task: mean_score, ... },
                    "by_time_of_day": { tod:  mean_score, ... },
                },
            }
    """
    _validate(df)

    scores = productivity_score(df)

    score_summary: dict = {
        "mean":   round(float(scores.mean()), 4),
        "median": round(float(scores.median()), 4),
        "std":    round(float(scores.std()), 4),
        "min":    round(float(scores.min()), 4),
        "max":    round(float(scores.max()), 4),
        "by_task_type": (
            scores.groupby(df["task_type"]).mean().round(4).to_dict()
        ),
        "by_time_of_day": (
            scores.groupby(df["time_of_day"]).mean().round(4).to_dict()
        ),
    }

    return {
        "completion_rate_by_time_of_day": completion_rate_by_time_of_day(df),
        "completion_rate_by_task_type":   completion_rate_by_task_type(df),
        "correlations":                   correlations(df),
        "productivity_score":             score_summary,
    }
