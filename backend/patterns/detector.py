"""
patterns/detector.py
---------------------
Pattern detection functions for the productivity dataset.

Each function accepts a pandas DataFrame (matching data/productivity.csv) and
returns a dict with:
  - "insight"  : a single human-readable sentence summarising the finding
  - "details"  : supporting numbers so the insight can be verified or explored
"""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Energy levels 4-5 are considered "high"
_HIGH_ENERGY_THRESHOLD = 4

# Focus levels 4-5 are considered "high" for flow-state detection
_HIGH_FOCUS_THRESHOLD = 4

# Distraction buckets (count per session)
_DISTRACTION_BUCKETS = {
    "none (0)":     (0,  0),
    "low (1-3)":    (1,  3),
    "medium (4-7)": (4,  7),
    "high (8+)":    (8,  999),
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate(df: pd.DataFrame) -> None:
    required = {
        "completed", "time_of_day", "task_type",
        "energy_level", "focus_level", "distraction_count",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def _pct(value: float) -> str:
    """Format a 0-1 rate as a percentage string, e.g. 0.477 → '47.7%'."""
    return f"{value * 100:.1f}%"


def _rank(series: pd.Series, ascending: bool = False) -> list[dict]:
    """Return a list of {label, rate} dicts sorted best-first."""
    sorted_s = series.sort_values(ascending=ascending)
    return [{"label": k, "rate": round(v, 4)} for k, v in sorted_s.items()]


# ---------------------------------------------------------------------------
# Public pattern-detection functions
# ---------------------------------------------------------------------------

def best_time_of_day(df: pd.DataFrame) -> dict:
    """
    Identify which time-of-day slot produces the highest completion rate and
    the highest average productivity score (energy + focus − distractions).

    Returns
    -------
    dict
        {
            "insight": "Morning is the most productive time of day, with a
                        47.7% completion rate — 19.0 pp ahead of afternoon
                        and 30.8 pp ahead of night.",
            "details": {
                "completion_rates":       { morning: 0.477, ... },
                "avg_focus_by_tod":       { morning: 3.4,  ... },
                "avg_energy_by_tod":      { morning: 3.8,  ... },
                "best_slot":              "morning",
                "best_completion_rate":   0.477,
                "runner_up_slot":         "afternoon",
                "gap_vs_runner_up_pp":    19.0,
                "gap_vs_worst_pp":        30.8,
            }
        }
    """
    _validate(df)

    completion = df.groupby("time_of_day")["completed"].mean().round(4)
    avg_focus  = df.groupby("time_of_day")["focus_level"].mean().round(2)
    avg_energy = df.groupby("time_of_day")["energy_level"].mean().round(2)

    ranked = completion.sort_values(ascending=False)
    best        = ranked.index[0]
    best_rate   = ranked.iloc[0]
    runner_up   = ranked.index[1] if len(ranked) > 1 else None
    worst       = ranked.index[-1]

    gap_runner_up = round((best_rate - ranked.iloc[1]) * 100, 1) if runner_up else 0.0
    gap_worst     = round((best_rate - ranked.iloc[-1]) * 100, 1)

    if runner_up:
        insight = (
            f"{best.capitalize()} is the most productive time of day, with a "
            f"{_pct(best_rate)} completion rate — {gap_runner_up} percentage "
            f"points ahead of {runner_up} and {gap_worst} pp ahead of {worst}."
        )
    else:
        insight = (
            f"{best.capitalize()} is the most productive time of day, with a "
            f"{_pct(best_rate)} completion rate."
        )

    return {
        "insight": insight,
        "details": {
            "completion_rates":     completion.to_dict(),
            "avg_focus_by_tod":     avg_focus.to_dict(),
            "avg_energy_by_tod":    avg_energy.to_dict(),
            "best_slot":            best,
            "best_completion_rate": best_rate,
            "runner_up_slot":       runner_up,
            "gap_vs_runner_up_pp":  gap_runner_up,
            "gap_vs_worst_pp":      gap_worst,
        },
    }


def best_task_type_when_energy_high(df: pd.DataFrame) -> dict:
    """
    Among sessions where energy is high (≥ 4), find which task type achieves
    the best completion rate and compare it to low-energy sessions.

    Returns
    -------
    dict
        {
            "insight": "When energy is high, deep_work has the best completion
                        rate at 71.4% — 2.1x higher than during low-energy
                        deep_work sessions.",
            "details": {
                "high_energy_threshold":          4,
                "high_energy_session_count":      142,
                "completion_by_task_high_energy": { deep_work: 0.714, ... },
                "completion_by_task_low_energy":  { deep_work: 0.333, ... },
                "best_task":                      "deep_work",
                "best_rate_high_energy":          0.714,
                "multiplier_vs_low_energy":       2.1,
            }
        }
    """
    _validate(df)

    high = df[df["energy_level"] >= _HIGH_ENERGY_THRESHOLD]
    low  = df[df["energy_level"] <  _HIGH_ENERGY_THRESHOLD]

    high_rates = high.groupby("task_type")["completed"].mean().round(4)
    low_rates  = low.groupby("task_type")["completed"].mean().round(4)

    best_task      = high_rates.idxmax()
    best_rate_high = high_rates[best_task]
    low_rate       = low_rates.get(best_task, None)

    if low_rate and low_rate > 0:
        multiplier = round(best_rate_high / low_rate, 1)
        mult_str   = f"{multiplier}x higher than during low-energy {best_task} sessions"
    else:
        multiplier = None
        mult_str   = "no comparable low-energy sessions found"

    insight = (
        f"When energy is high (≥{_HIGH_ENERGY_THRESHOLD}), {best_task.replace('_', ' ')} "
        f"has the best completion rate at {_pct(best_rate_high)} — {mult_str}."
    )

    return {
        "insight": insight,
        "details": {
            "high_energy_threshold":          _HIGH_ENERGY_THRESHOLD,
            "high_energy_session_count":      int(len(high)),
            "completion_by_task_high_energy": high_rates.to_dict(),
            "completion_by_task_low_energy":  low_rates.to_dict(),
            "best_task":                      best_task,
            "best_rate_high_energy":          best_rate_high,
            "multiplier_vs_low_energy":       multiplier,
        },
    }


def distraction_impact(df: pd.DataFrame) -> dict:
    """
    Measure how distraction count affects completion rate and average focus,
    bucketed into none / low / medium / high bands.

    Returns
    -------
    dict
        {
            "insight": "Sessions with no distractions complete at 61.5%,
                        dropping to 14.3% when distractions are high (8+) —
                        a 47.2 pp difference.",
            "details": {
                "buckets": {
                    "none (0)":     { "session_count": 45, "completion_rate": 0.615,
                                      "avg_focus": 4.1, "avg_distraction_duration": 0.0 },
                    ...
                },
                "best_bucket":          "none (0)",
                "worst_bucket":         "high (8+)",
                "completion_drop_pp":   47.2,
                "correlation_r":        -0.296,
            }
        }
    """
    _validate(df)

    bucket_stats: dict[str, dict] = {}
    for label, (lo, hi) in _DISTRACTION_BUCKETS.items():
        subset = df[(df["distraction_count"] >= lo) & (df["distraction_count"] <= hi)]
        if subset.empty:
            continue
        bucket_stats[label] = {
            "session_count":            int(len(subset)),
            "completion_rate":          round(float(subset["completed"].mean()), 4),
            "avg_focus":                round(float(subset["focus_level"].mean()), 2),
            "avg_distraction_duration": round(float(subset["distraction_duration"].mean()), 1)
            if "distraction_duration" in df.columns else None,
        }

    rates = {k: v["completion_rate"] for k, v in bucket_stats.items()}
    best_bucket  = max(rates, key=rates.get)
    worst_bucket = min(rates, key=rates.get)
    drop_pp      = round((rates[best_bucket] - rates[worst_bucket]) * 100, 1)
    corr_r       = round(float(df["distraction_count"].corr(df["completed"])), 3)

    insight = (
        f"Sessions with {best_bucket} distractions complete at "
        f"{_pct(rates[best_bucket])}, dropping to {_pct(rates[worst_bucket])} "
        f"when distractions are {worst_bucket} — a {drop_pp} percentage point "
        f"difference (r = {corr_r})."
    )

    return {
        "insight": insight,
        "details": {
            "buckets":            bucket_stats,
            "best_bucket":        best_bucket,
            "worst_bucket":       worst_bucket,
            "completion_drop_pp": drop_pp,
            "correlation_r":      corr_r,
        },
    }


def flow_state_analysis(df: pd.DataFrame) -> dict:
    """
    Detect "flow state" sessions — defined as focus_level ≥ 4 AND completed == 1.
    Report how common flow is, which conditions produce it most, and how flow
    sessions compare to non-flow sessions on key metrics.

    Returns
    -------
    dict
        {
            "insight": "Flow state (high focus + task completed) occurs in 22.5%
                        of all sessions. It is most common during morning deep_work
                        sessions (48.3%), where average energy is 4.1.",
            "details": {
                "flow_definition":          "focus_level >= 4 AND completed == 1",
                "total_sessions":           400,
                "flow_sessions":            90,
                "flow_rate_overall":        0.225,
                "flow_rate_by_tod":         { morning: 0.35, ... },
                "flow_rate_by_task":        { deep_work: 0.38, ... },
                "top_condition":            "morning + deep_work",
                "top_condition_flow_rate":  0.483,
                "avg_energy_in_flow":       4.1,
                "avg_energy_not_in_flow":   2.8,
                "avg_distractions_in_flow": 2.3,
                "avg_distractions_not_in_flow": 5.9,
            }
        }
    """
    _validate(df)

    flow_mask    = (df["focus_level"] >= _HIGH_FOCUS_THRESHOLD) & (df["completed"] == 1)
    flow_df      = df[flow_mask]
    non_flow_df  = df[~flow_mask]

    total        = len(df)
    flow_count   = int(flow_mask.sum())
    flow_rate    = round(flow_count / total, 4) if total else 0.0

    flow_by_tod  = (
        df.groupby("time_of_day")
        .apply(lambda g: round((
            (g["focus_level"] >= _HIGH_FOCUS_THRESHOLD) & (g["completed"] == 1)
        ).mean(), 4), include_groups=False)
        .to_dict()
    )
    flow_by_task = (
        df.groupby("task_type")
        .apply(lambda g: round((
            (g["focus_level"] >= _HIGH_FOCUS_THRESHOLD) & (g["completed"] == 1)
        ).mean(), 4), include_groups=False)
        .to_dict()
    )

    # Best tod × task_type combination
    combo = (
        df.groupby(["time_of_day", "task_type"])
        .apply(lambda g: (
            (g["focus_level"] >= _HIGH_FOCUS_THRESHOLD) & (g["completed"] == 1)
        ).mean(), include_groups=False)
        .round(4)
    )
    top_combo       = combo.idxmax()          # (tod, task_type) tuple
    top_combo_rate  = round(float(combo.max()), 4)
    top_label       = f"{top_combo[0]} + {top_combo[1].replace('_', ' ')}"

    avg_energy_flow     = round(float(flow_df["energy_level"].mean()), 2)    if flow_count else 0.0
    avg_energy_nflow    = round(float(non_flow_df["energy_level"].mean()), 2) if len(non_flow_df) else 0.0
    avg_dist_flow       = round(float(flow_df["distraction_count"].mean()), 2)    if flow_count else 0.0
    avg_dist_nflow      = round(float(non_flow_df["distraction_count"].mean()), 2) if len(non_flow_df) else 0.0

    insight = (
        f"Flow state (focus ≥ {_HIGH_FOCUS_THRESHOLD} and task completed) occurs in "
        f"{_pct(flow_rate)} of all sessions. It is most common during "
        f"{top_label} sessions ({_pct(top_combo_rate)}), where average energy "
        f"is {avg_energy_flow} — versus {avg_energy_nflow} outside flow state."
    )

    return {
        "insight": insight,
        "details": {
            "flow_definition":              f"focus_level >= {_HIGH_FOCUS_THRESHOLD} AND completed == 1",
            "total_sessions":               total,
            "flow_sessions":                flow_count,
            "flow_rate_overall":            flow_rate,
            "flow_rate_by_tod":             flow_by_tod,
            "flow_rate_by_task":            flow_by_task,
            "top_condition":                top_label,
            "top_condition_flow_rate":      top_combo_rate,
            "avg_energy_in_flow":           avg_energy_flow,
            "avg_energy_not_in_flow":       avg_energy_nflow,
            "avg_distractions_in_flow":     avg_dist_flow,
            "avg_distractions_not_in_flow": avg_dist_nflow,
        },
    }


def all_patterns(df: pd.DataFrame) -> dict:
    """
    Run all four pattern detectors and return a consolidated dict.

    Returns
    -------
    dict
        {
            "best_time_of_day":               { "insight": "...", "details": {...} },
            "best_task_when_energy_high":      { "insight": "...", "details": {...} },
            "distraction_impact":              { "insight": "...", "details": {...} },
            "flow_state":                      { "insight": "...", "details": {...} },
        }
    """
    _validate(df)
    return {
        "best_time_of_day":          best_time_of_day(df),
        "best_task_when_energy_high": best_task_type_when_energy_high(df),
        "distraction_impact":        distraction_impact(df),
        "flow_state":                flow_state_analysis(df),
    }
