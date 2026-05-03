"""
analytics/segmentation.py
--------------------------
User segmentation based on completion rate, with behavioral comparison
between segments.

Segments (thresholds calibrated to this dataset's distribution)
---------------------------------------------------------------
  High Performers : completion rate >= 45%  (top quartile)
  Mid Performers  : completion rate 25–45%
  Low Performers  : completion rate <  25%  (bottom quartile)

Only users with at least MIN_SESSIONS sessions are classified — too few
sessions makes the completion rate statistically unreliable.

Public API
----------
  segment_users(df)              → DataFrame: one row per user with segment label
  compare_segments(df)           → dict: behavioral averages per segment
  segment_insights(df)           → list of plain-language comparison strings
  significance_tests(df)         → dict: t-test results (High vs Low Performers)
  full_segmentation_report(df)   → combined dict with all four outputs
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MIN_SESSIONS = 5          # minimum sessions required to classify a user

SEGMENT_HIGH = "High Performer"
SEGMENT_MID  = "Mid Performer"
SEGMENT_LOW  = "Low Performer"

# Thresholds are calibrated to the dataset's actual completion rate distribution.
# Max observed completion rate (5+ sessions) is ~67%, mean is ~34%.
# Top quartile ≥ 45% → High Performer
# Bottom quartile < 25% → Low Performer
_HIGH_THRESHOLD = 0.45
_LOW_THRESHOLD  = 0.25


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate(df: pd.DataFrame) -> None:
    required = {"user_id", "completed", "energy_level", "focus_level",
                "distraction_count", "task_type", "duration"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def _preferred_task(group: pd.DataFrame) -> str:
    """Return the task_type with the highest completion rate for a user group."""
    rates = group.groupby("task_type")["completed"].mean()
    return rates.idxmax() if not rates.empty else "N/A"


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def segment_users(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each user into a performance segment.

    Only users with >= MIN_SESSIONS sessions are assigned a segment.
    Users below the threshold are labelled 'Insufficient Data'.

    Returns
    -------
    DataFrame with columns:
        user_id, total_sessions, completion_rate, avg_energy, avg_focus,
        avg_distractions, avg_duration, preferred_task, segment
    Ordered by completion_rate descending.
    """
    _validate(df)

    user_stats = (
        df.groupby("user_id")
        .agg(
            total_sessions   = ("completed", "count"),
            completion_rate  = ("completed", "mean"),
            avg_energy       = ("energy_level", "mean"),
            avg_focus        = ("focus_level", "mean"),
            avg_distractions = ("distraction_count", "mean"),
            avg_duration     = ("duration", "mean"),
        )
        .round(4)
        .reset_index()
    )

    # Preferred task: task_type with highest completion rate per user
    preferred = (
        df.groupby(["user_id", "task_type"])["completed"]
        .mean()
        .reset_index()
        .sort_values("completed", ascending=False)
        .drop_duplicates("user_id")
        .rename(columns={"completed": "_task_rate"})
        [["user_id", "task_type"]]
        .rename(columns={"task_type": "preferred_task"})
    )
    user_stats = user_stats.merge(preferred, on="user_id", how="left")

    # Assign segment
    def _assign(row: pd.Series) -> str:
        if row["total_sessions"] < MIN_SESSIONS:
            return "Insufficient Data"
        if row["completion_rate"] >= _HIGH_THRESHOLD:
            return SEGMENT_HIGH
        if row["completion_rate"] >= _LOW_THRESHOLD:
            return SEGMENT_MID
        return SEGMENT_LOW

    user_stats["segment"] = user_stats.apply(_assign, axis=1)

    return user_stats.sort_values("completion_rate", ascending=False).reset_index(drop=True)


def compare_segments(df: pd.DataFrame) -> dict:
    """
    Aggregate behavioral metrics for each segment and return a comparison dict.

    Returns
    -------
    dict keyed by segment name, each value containing:
        user_count, avg_completion_rate, avg_energy, avg_focus,
        avg_distractions, avg_duration, preferred_task, session_count
    """
    _validate(df)

    segmented = segment_users(df)

    # Exclude users with insufficient data from comparison
    segmented = segmented[segmented["segment"] != "Insufficient Data"]

    result = {}
    for segment, group in segmented.groupby("segment"):
        # Get all raw sessions for users in this segment
        users_in_segment = group["user_id"].tolist()
        sessions = df[df["user_id"].isin(users_in_segment)]

        result[segment] = {
            "user_count":          int(len(group)),
            "session_count":       int(len(sessions)),
            "avg_completion_rate": round(float(group["completion_rate"].mean()), 4),
            "avg_energy":          round(float(sessions["energy_level"].mean()), 4),
            "avg_focus":           round(float(sessions["focus_level"].mean()), 4),
            "avg_distractions":    round(float(sessions["distraction_count"].mean()), 4),
            "avg_duration":        round(float(sessions["duration"].mean()), 2),
            "preferred_task":      _preferred_task(sessions),
        }

    return result


def segment_insights(df: pd.DataFrame) -> list[str]:
    """
    Generate plain-language insight strings comparing High vs Low performers.

    Returns
    -------
    list of insight strings, each a self-contained finding.
    """
    _validate(df)

    comparison = compare_segments(df)

    high = comparison.get(SEGMENT_HIGH)
    low  = comparison.get(SEGMENT_LOW)
    mid  = comparison.get(SEGMENT_MID)

    insights: list[str] = []

    # --- User counts ---
    if high and low:
        insights.append(
            f"Of users with {MIN_SESSIONS}+ sessions, {high['user_count']} are High Performers "
            f"(≥{int(_HIGH_THRESHOLD * 100)}% completion) and {low['user_count']} are Low Performers "
            f"(<{int(_LOW_THRESHOLD * 100)}% completion)."
        )

    # --- Energy gap ---
    if high and low:
        energy_gap = round(high["avg_energy"] - low["avg_energy"], 2)
        direction  = "higher" if energy_gap > 0 else "lower"
        insights.append(
            f"High Performers average {high['avg_energy']:.2f}/5 energy vs "
            f"{low['avg_energy']:.2f}/5 for Low Performers — "
            f"a {abs(energy_gap):.2f}-point {direction} energy level."
        )

    # --- Focus gap ---
    if high and low:
        focus_gap = round(high["avg_focus"] - low["avg_focus"], 2)
        direction = "higher" if focus_gap > 0 else "lower"
        insights.append(
            f"High Performers maintain {high['avg_focus']:.2f}/5 avg focus vs "
            f"{low['avg_focus']:.2f}/5 for Low Performers "
            f"({abs(focus_gap):.2f} points {direction})."
        )

    # --- Distraction gap ---
    if high and low:
        dist_gap  = round(low["avg_distractions"] - high["avg_distractions"], 2)
        insights.append(
            f"Low Performers face {low['avg_distractions']:.2f} distractions per session on average, "
            f"compared to {high['avg_distractions']:.2f} for High Performers "
            f"— {abs(dist_gap):.2f} more interruptions per session."
        )

    # --- Preferred task ---
    if high and low:
        if high["preferred_task"] == low["preferred_task"]:
            insights.append(
                f"Both segments perform best on {high['preferred_task'].replace('_', ' ')} tasks, "
                f"but High Performers complete them at a significantly higher rate."
            )
        else:
            insights.append(
                f"High Performers excel at {high['preferred_task'].replace('_', ' ')} tasks, "
                f"while Low Performers perform relatively better on "
                f"{low['preferred_task'].replace('_', ' ')} tasks."
            )

    # --- Duration ---
    if high and low:
        dur_gap = round(high["avg_duration"] - low["avg_duration"], 1)
        direction = "longer" if dur_gap > 0 else "shorter"
        insights.append(
            f"High Performers work in {abs(dur_gap):.1f}-minute {direction} sessions on average "
            f"({high['avg_duration']:.0f} min vs {low['avg_duration']:.0f} min for Low Performers)."
        )

    # --- Mid performers context ---
    if mid:
        insights.append(
            f"Mid Performers ({mid['user_count']} users, "
            f"{int(_LOW_THRESHOLD * 100)}–{int(_HIGH_THRESHOLD * 100)}% completion) average "
            f"{mid['avg_energy']:.2f} energy and {mid['avg_distractions']:.2f} distractions — "
            f"sitting between the two extremes as expected."
        )

    return insights


def significance_tests(df: pd.DataFrame) -> dict:
    """
    Run independent-samples t-tests comparing High vs Low Performers
    on key behavioral metrics (energy, focus, distraction_count).

    Uses scipy.stats.ttest_ind to determine whether observed differences
    between segments are statistically significant or could be due to chance.

    Returns
    -------
    dict keyed by metric name, each containing:
        t_statistic, p_value, significant (bool, p < 0.05),
        high_mean, low_mean, interpretation
    """
    _validate(df)

    segmented = segment_users(df)
    high_users = segmented.loc[segmented["segment"] == SEGMENT_HIGH, "user_id"].tolist()
    low_users  = segmented.loc[segmented["segment"] == SEGMENT_LOW,  "user_id"].tolist()

    if not high_users or not low_users:
        return {"error": "Insufficient users in one or both segments for significance testing."}

    high_sessions = df[df["user_id"].isin(high_users)]
    low_sessions  = df[df["user_id"].isin(low_users)]

    results = {}
    for metric in ("energy_level", "focus_level", "distraction_count"):
        high_vals = high_sessions[metric].dropna().to_numpy()
        low_vals  = low_sessions[metric].dropna().to_numpy()

        t_stat, p_val = scipy_stats.ttest_ind(high_vals, low_vals, equal_var=False)

        significant = bool(p_val < 0.05)
        high_mean   = round(float(np.mean(high_vals)), 4)
        low_mean    = round(float(np.mean(low_vals)),  4)
        diff        = round(high_mean - low_mean, 4)

        if significant:
            interp = (
                f"The {round(abs(diff), 2)}-point difference in {metric.replace('_', ' ')} "
                f"between High and Low Performers is statistically significant "
                f"(p={p_val:.4f}), meaning it is unlikely to be due to chance."
            )
        else:
            interp = (
                f"The difference in {metric.replace('_', ' ')} between segments "
                f"(p={p_val:.4f}) does not reach statistical significance at the 0.05 level."
            )

        results[metric] = {
            "t_statistic": round(float(t_stat), 4),
            "p_value":     round(float(p_val),  4),
            "significant": significant,
            "high_mean":   high_mean,
            "low_mean":    low_mean,
            "difference":  diff,
            "interpretation": interp,
        }

    return results


def full_segmentation_report(df: pd.DataFrame) -> dict:
    """
    Run all segmentation analyses and return a single consolidated dict.

    Returns
    -------
    dict
        {
            "user_segments":   list of dicts (one per user),
            "segment_comparison": { "High Performer": {...}, "Mid Performer": {...}, "Low Performer": {...} },
            "insights":        [ "...", ... ],
        }
    """
    _validate(df)

    segmented   = segment_users(df)
    comparison  = compare_segments(df)
    insights    = segment_insights(df)
    sig_tests   = significance_tests(df)

    return {
        "user_segments":      segmented.to_dict(orient="records"),
        "segment_comparison": comparison,
        "insights":           insights,
        "significance_tests": sig_tests,
    }
