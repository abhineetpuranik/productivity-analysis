"""
scoring/engine.py
------------------
Per-row scoring logic for the productivity dataset.

Two scores are defined:

  Productivity Index (0–100)
  --------------------------
  A holistic measure of how productive a session was, combining task
  completion, energy, focus, and distraction penalty into a single
  0–100 integer-friendly scale.

      PI = 100 × (
              0.35 × completed
            + 0.25 × norm(energy_level,      1, 5)
            + 0.25 × norm(focus_level,        1, 5)
            + 0.15 × (1 − norm(distraction_count, 0, 15))
           )

  Cognitive Load Score (0–100)
  ----------------------------
  Estimates the mental effort demanded by a session based on task type,
  session duration, and the distraction overhead the user had to manage.

  Task-type base load (reflects inherent cognitive demand):
      deep_work  → 70   (sustained, high-focus effort)
      creative   → 55   (generative, moderately demanding)
      admin      → 35   (routine, lower executive demand)

  Duration modifier (+0–15 pts):
      Longer sessions add up to 15 points, scaled against a 180-minute cap.
      This reflects fatigue accumulation over time.

  Distraction overhead (+0–15 pts):
      Each distraction interrupts working memory. Scaled against a 15-count cap.

      CL = base_load
         + 15 × norm(duration,          0, 180)
         + 15 × norm(distraction_count, 0,  15)

  Both scores are clipped to [0, 100] and rounded to 2 decimal places.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

# Productivity Index weights (must sum to 1.0)
_PI_W_COMPLETION   = 0.35
_PI_W_ENERGY       = 0.25
_PI_W_FOCUS        = 0.25
_PI_W_DISTRACTION  = 0.15   # penalty (inverted)

# Feature bounds for normalisation
_ENERGY_MIN,      _ENERGY_MAX      = 1,   5
_FOCUS_MIN,       _FOCUS_MAX       = 1,   5
_DISTRACTION_MIN, _DISTRACTION_MAX = 0,  15
_DURATION_MIN,    _DURATION_MAX    = 0, 180

# Cognitive Load base scores per task type
_CL_BASE: dict[str, float] = {
    "deep_work": 70.0,
    "creative":  55.0,
    "admin":     35.0,
}
_CL_DEFAULT_BASE = 50.0   # fallback for unknown task types

# Cognitive Load modifier caps (points added on top of base)
_CL_DURATION_CAP     = 15.0
_CL_DISTRACTION_CAP  = 15.0

# Label bands for both scores
_PI_BANDS: list[tuple[float, str]] = [
    (80, "Excellent"),
    (60, "Good"),
    (40, "Moderate"),
    (20, "Low"),
    ( 0, "Poor"),
]

_CL_BANDS: list[tuple[float, str]] = [
    (80, "Very High"),
    (65, "High"),
    (50, "Moderate"),
    (35, "Low"),
    ( 0, "Minimal"),
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate(df: pd.DataFrame) -> None:
    required = {
        "completed", "task_type", "duration",
        "energy_level", "focus_level", "distraction_count",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def _norm(series: pd.Series, lo: float, hi: float) -> pd.Series:
    """Min-max normalise to [0, 1] using fixed domain bounds."""
    return (series.clip(lo, hi) - lo) / (hi - lo)


def _band_label(score: float, bands: list[tuple[float, str]]) -> str:
    """Return the human-readable band label for a given score."""
    for threshold, label in bands:
        if score >= threshold:
            return label
    return bands[-1][1]


def _summary_stats(series: pd.Series) -> dict[str, float]:
    return {
        "mean":   round(float(series.mean()),   2),
        "median": round(float(series.median()), 2),
        "std":    round(float(series.std()),    2),
        "min":    round(float(series.min()),    2),
        "max":    round(float(series.max()),    2),
    }


def _percentile_stats(series: pd.Series) -> dict[str, float]:
    """
    Compute percentile distribution using numpy.
    Gives a fuller picture of score spread than mean/median alone.
    """
    arr = series.dropna().to_numpy()
    return {
        "p10": round(float(np.percentile(arr, 10)), 2),
        "p25": round(float(np.percentile(arr, 25)), 2),
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p75": round(float(np.percentile(arr, 75)), 2),
        "p90": round(float(np.percentile(arr, 90)), 2),
    }


# ---------------------------------------------------------------------------
# Public scoring functions
# ---------------------------------------------------------------------------

def productivity_index(df: pd.DataFrame) -> pd.Series:
    """
    Compute the Productivity Index (0–100) for every row.

    Formula
    -------
    ::

        PI = 100 × (
                0.35 × completed
              + 0.25 × norm(energy_level,      1,  5)
              + 0.25 × norm(focus_level,        1,  5)
              + 0.15 × (1 − norm(distraction_count, 0, 15))
             )

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``completed``, ``energy_level``, ``focus_level``,
        ``distraction_count``.

    Returns
    -------
    pd.Series
        Float scores in [0, 100], rounded to 2 dp, named
        ``"productivity_index"``.
    """
    _validate(df)

    raw = (
        _PI_W_COMPLETION  * df["completed"]
        + _PI_W_ENERGY    * _norm(df["energy_level"],      _ENERGY_MIN,      _ENERGY_MAX)
        + _PI_W_FOCUS     * _norm(df["focus_level"],       _FOCUS_MIN,       _FOCUS_MAX)
        + _PI_W_DISTRACTION * (1 - _norm(df["distraction_count"], _DISTRACTION_MIN, _DISTRACTION_MAX))
    )

    return (raw * 100).clip(0, 100).round(2).rename("productivity_index")


def cognitive_load_score(df: pd.DataFrame) -> pd.Series:
    """
    Compute the Cognitive Load Score (0–100) for every row.

    Formula
    -------
    ::

        base     = task_type base load  (deep_work=70, creative=55, admin=35)
        duration_mod     = 15 × norm(duration,          0, 180)
        distraction_mod  = 15 × norm(distraction_count, 0,  15)

        CL = base + duration_mod + distraction_mod   (clipped to [0, 100])

    A higher score means the session demanded more cognitive effort.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``task_type``, ``duration``, ``distraction_count``.

    Returns
    -------
    pd.Series
        Float scores in [0, 100], rounded to 2 dp, named
        ``"cognitive_load_score"``.
    """
    _validate(df)

    base = df["task_type"].map(_CL_BASE).fillna(_CL_DEFAULT_BASE)

    duration_mod    = _CL_DURATION_CAP    * _norm(df["duration"],           _DURATION_MIN,    _DURATION_MAX)
    distraction_mod = _CL_DISTRACTION_CAP * _norm(df["distraction_count"],  _DISTRACTION_MIN, _DISTRACTION_MAX)

    cl = (base + duration_mod + distraction_mod).clip(0, 100).round(2)
    return cl.rename("cognitive_load_score")


def score_dataset(df: pd.DataFrame) -> dict:
    """
    Score every row in the dataset and return a comprehensive results dict.

    The returned dict contains:

    - ``"scores"``        — the original DataFrame with two new columns appended:
                            ``productivity_index`` and ``cognitive_load_score``
    - ``"pi_summary"``    — descriptive stats + band breakdown for Productivity Index
    - ``"cl_summary"``    — descriptive stats + band breakdown for Cognitive Load Score
    - ``"by_task_type"``  — mean PI and CL for each task type
    - ``"by_time_of_day"``— mean PI and CL for each time-of-day slot (if column present)
    - ``"score_weights"`` — the formula weights used, for transparency

    Parameters
    ----------
    df : pd.DataFrame
        Full productivity DataFrame.

    Returns
    -------
    dict
    """
    _validate(df)

    scored = df.copy()
    scored["productivity_index"]   = productivity_index(df)
    scored["cognitive_load_score"] = cognitive_load_score(df)

    # --- Band breakdowns ---
    def _band_breakdown(series: pd.Series, bands: list[tuple[float, str]]) -> dict[str, int]:
        labels = series.apply(lambda v: _band_label(v, bands))
        return labels.value_counts().to_dict()

    pi_summary: dict = {
        **_summary_stats(scored["productivity_index"]),
        "percentiles": _percentile_stats(scored["productivity_index"]),
        "band_counts": _band_breakdown(scored["productivity_index"], _PI_BANDS),
        "band_definition": {label: f">= {thr}" for thr, label in _PI_BANDS},
    }

    cl_summary: dict = {
        **_summary_stats(scored["cognitive_load_score"]),
        "percentiles": _percentile_stats(scored["cognitive_load_score"]),
        "band_counts": _band_breakdown(scored["cognitive_load_score"], _CL_BANDS),
        "band_definition": {label: f">= {thr}" for thr, label in _CL_BANDS},
        "base_loads": _CL_BASE,
    }

    # --- Breakdowns by task type ---
    by_task = (
        scored.groupby("task_type")[["productivity_index", "cognitive_load_score"]]
        .mean()
        .round(2)
        .rename(columns={
            "productivity_index":   "mean_productivity_index",
            "cognitive_load_score": "mean_cognitive_load_score",
        })
        .to_dict(orient="index")
    )

    # --- Breakdowns by time of day (optional column) ---
    by_tod: dict | None = None
    if "time_of_day" in df.columns:
        by_tod = (
            scored.groupby("time_of_day")[["productivity_index", "cognitive_load_score"]]
            .mean()
            .round(2)
            .rename(columns={
                "productivity_index":   "mean_productivity_index",
                "cognitive_load_score": "mean_cognitive_load_score",
            })
            .to_dict(orient="index")
        )

    return {
        "scores":         scored,
        "pi_summary":     pi_summary,
        "cl_summary":     cl_summary,
        "by_task_type":   by_task,
        "by_time_of_day": by_tod,
        "score_weights": {
            "productivity_index": {
                "completed":         _PI_W_COMPLETION,
                "energy_level":      _PI_W_ENERGY,
                "focus_level":       _PI_W_FOCUS,
                "distraction_count": f"-{_PI_W_DISTRACTION} (penalty)",
            },
            "cognitive_load_score": {
                "task_type_base":    _CL_BASE,
                "duration_modifier": f"up to +{_CL_DURATION_CAP} pts",
                "distraction_modifier": f"up to +{_CL_DISTRACTION_CAP} pts",
            },
        },
    }
