"""
analytics/recommendations.py
------------------------------
Data-driven recommendations generated entirely from computed analysis results.
No fixed rules — every recommendation is derived from the actual dataset.

Three core computations
-----------------------
1. Best time_of_day        : time slot with the highest completion rate
2. Best task_type per energy level : for each energy band (low/medium/high),
                             which task_type yields the best completion rate
3. Distraction threshold   : the distraction_count level at which completion
                             rate drops below a meaningful performance floor

Public API
----------
  best_time_of_day(df)              → dict with the winning slot and supporting stats
  best_task_by_energy(df)           → dict mapping energy band → best task_type
  distraction_threshold(df)         → dict with threshold value and performance cliff
  generate_recommendations(df)      → list of actionable insight strings
  full_recommendations_report(df)   → combined dict with all outputs
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Energy bands (based on 1–5 scale)
_ENERGY_BANDS = {
    "low":    (1, 2),
    "medium": (3, 3),
    "high":   (4, 5),
}

# Minimum sessions required for a group to be considered statistically reliable
_MIN_GROUP_SIZE = 10

# Performance floor: completion rate below this is considered a significant drop
_PERFORMANCE_FLOOR = 0.25


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate(df: pd.DataFrame) -> None:
    required = {
        "completed", "time_of_day", "task_type",
        "energy_level", "distraction_count",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


# ---------------------------------------------------------------------------
# 1. Best time of day
# ---------------------------------------------------------------------------

def best_time_of_day(df: pd.DataFrame) -> dict:
    """
    Identify the time_of_day slot with the highest completion rate.

    Returns
    -------
    dict
        {
            "best_slot":        str,
            "completion_rates": { slot: rate, ... },   # all slots ranked
            "lift_over_worst":  float,                 # percentage point gap
            "session_counts":   { slot: count, ... },
        }
    """
    _validate(df)

    grouped = df.groupby("time_of_day").agg(
        completion_rate=("completed", "mean"),
        session_count=("completed", "count"),
    ).round(4)

    rates  = grouped["completion_rate"].to_dict()
    counts = grouped["session_count"].to_dict()

    best  = max(rates, key=rates.get)
    worst = min(rates, key=rates.get)
    lift  = round(rates[best] - rates[worst], 4)

    # Rank slots best → worst
    ranked = dict(sorted(rates.items(), key=lambda x: x[1], reverse=True))

    return {
        "best_slot":        best,
        "completion_rates": ranked,
        "lift_over_worst":  lift,
        "session_counts":   counts,
    }


# ---------------------------------------------------------------------------
# 2. Best task type per energy level
# ---------------------------------------------------------------------------

def best_task_by_energy(df: pd.DataFrame) -> dict:
    """
    For each energy band (low / medium / high), find the task_type with the
    highest completion rate.

    Returns
    -------
    dict
        {
            "low":    { "best_task": str, "completion_rate": float, "session_count": int },
            "medium": { ... },
            "high":   { ... },
            "all_rates": { band: { task_type: rate } }   # full breakdown
        }
    """
    _validate(df)

    result     = {}
    all_rates  = {}

    for band, (lo, hi) in _ENERGY_BANDS.items():
        subset = df[df["energy_level"].between(lo, hi)]

        if len(subset) < _MIN_GROUP_SIZE:
            result[band]    = {"best_task": "N/A", "completion_rate": None, "session_count": len(subset)}
            all_rates[band] = {}
            continue

        task_rates = (
            subset.groupby("task_type")["completed"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "rate", "count": "n"})
        )

        # Only consider task types with enough sessions
        reliable = task_rates[task_rates["n"] >= _MIN_GROUP_SIZE]
        if reliable.empty:
            reliable = task_rates  # fall back to all if none meet threshold

        best_task = reliable["rate"].idxmax()
        best_rate = round(float(reliable.loc[best_task, "rate"]), 4)
        best_n    = int(reliable.loc[best_task, "n"])

        result[band] = {
            "best_task":       best_task,
            "completion_rate": best_rate,
            "session_count":   best_n,
        }
        all_rates[band] = task_rates["rate"].round(4).to_dict()

    result["all_rates"] = all_rates
    return result


# ---------------------------------------------------------------------------
# 3. Distraction threshold
# ---------------------------------------------------------------------------

def distraction_threshold(df: pd.DataFrame) -> dict:
    """
    Find the distraction_count level at which completion rate drops below
    the performance floor (_PERFORMANCE_FLOOR).

    Uses a rolling average over distraction count buckets to smooth noise,
    then finds the first bucket where performance falls below the floor.

    Returns
    -------
    dict
        {
            "threshold":          int or None,   # distraction count where perf drops
            "performance_floor":  float,         # the floor used
            "rate_at_threshold":  float or None,
            "rate_below_floor":   float or None, # avg rate for sessions above threshold
            "rate_above_floor":   float or None, # avg rate for sessions below threshold
            "by_distraction_count": { count: rate },  # full per-count breakdown
        }
    """
    _validate(df)

    # Completion rate at each exact distraction count
    by_count = (
        df.groupby("distraction_count")["completed"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "rate", "count": "n"})
    )

    # Smooth with a simple rolling mean (window=2) to reduce single-count noise
    by_count["smoothed"] = by_count["rate"].rolling(window=2, min_periods=1).mean()

    full_breakdown = by_count["rate"].round(4).to_dict()

    # Find first distraction count where smoothed rate drops below floor
    threshold     = None
    rate_at_thr   = None

    for count, row in by_count.iterrows():
        if row["smoothed"] < _PERFORMANCE_FLOOR:
            threshold   = int(count)
            rate_at_thr = round(float(row["rate"]), 4)
            break

    # Compare sessions below vs above threshold
    if threshold is not None:
        below = df[df["distraction_count"] <  threshold]
        above = df[df["distraction_count"] >= threshold]
        rate_below = round(float(below["completed"].mean()), 4) if len(below) else None
        rate_above = round(float(above["completed"].mean()), 4) if len(above) else None
    else:
        rate_below = None
        rate_above = None

    return {
        "threshold":              threshold,
        "performance_floor":      _PERFORMANCE_FLOOR,
        "rate_at_threshold":      rate_at_thr,
        "rate_below_threshold":   rate_below,
        "rate_above_threshold":   rate_above,
        "by_distraction_count":   full_breakdown,
    }


# ---------------------------------------------------------------------------
# 4. Generate recommendations
# ---------------------------------------------------------------------------

def generate_recommendations(df: pd.DataFrame) -> list[str]:
    """
    Generate actionable recommendations derived entirely from computed values.
    No fixed rules — every statement references actual numbers from the data.

    Returns
    -------
    list of recommendation strings, each self-contained and actionable.
    """
    _validate(df)

    tod   = best_time_of_day(df)
    task  = best_task_by_energy(df)
    dist  = distraction_threshold(df)

    recs: list[str] = []

    # --- Time of day ---
    best_slot   = tod["best_slot"]
    best_rate   = tod["completion_rates"][best_slot]
    lift        = tod["lift_over_worst"]
    worst_slot  = min(tod["completion_rates"], key=tod["completion_rates"].get)
    worst_rate  = tod["completion_rates"][worst_slot]

    recs.append(
        f"Schedule your most important work in the {best_slot}: completion rate is "
        f"{_pct(best_rate)}, compared to {_pct(worst_rate)} at {worst_slot} — "
        f"a {_pct(lift)} gap driven entirely by time of day."
    )

    # --- Task type by energy ---
    high_band = task.get("high", {})
    low_band  = task.get("low", {})
    med_band  = task.get("medium", {})

    if high_band.get("best_task") and high_band.get("best_task") != "N/A":
        recs.append(
            f"When your energy is high (4–5/5), prioritise "
            f"{high_band['best_task'].replace('_', ' ')} tasks — they complete at "
            f"{_pct(high_band['completion_rate'])} under high energy conditions."
        )

    if low_band.get("best_task") and low_band.get("best_task") != "N/A":
        recs.append(
            f"When energy is low (1–2/5), switch to "
            f"{low_band['best_task'].replace('_', ' ')} tasks — the data shows "
            f"these still complete at {_pct(low_band['completion_rate'])} even in low-energy sessions."
        )

    if med_band.get("best_task") and med_band.get("best_task") != "N/A":
        recs.append(
            f"At medium energy (3/5), {med_band['best_task'].replace('_', ' ')} tasks "
            f"yield the best results with a {_pct(med_band['completion_rate'])} completion rate."
        )

    # --- Distraction threshold ---
    if dist["threshold"] is not None:
        thr         = dist["threshold"]
        rate_below  = dist["rate_below_threshold"]
        rate_above  = dist["rate_above_threshold"]

        recs.append(
            f"Keep distractions below {thr} per session. "
            f"Sessions with fewer than {thr} distractions complete at {_pct(rate_below)}, "
            f"but performance drops to {_pct(rate_above)} once that threshold is crossed."
        )
    else:
        # No hard cliff found — report the trend instead
        by_count = dist["by_distraction_count"]
        if by_count:
            counts  = sorted(by_count.keys())
            low_d   = round(float(np.mean([by_count[c] for c in counts[:3]])), 4)
            high_d  = round(float(np.mean([by_count[c] for c in counts[-3:]])), 4)
            recs.append(
                f"Minimise distractions: sessions with the fewest interruptions complete at "
                f"{_pct(low_d)} on average, vs {_pct(high_d)} for the most distracted sessions."
            )

    # --- Cross-cutting insight: energy × time alignment ---
    tod_energy = df.groupby("time_of_day")["energy_level"].mean().round(2)
    peak_energy_slot = tod_energy.idxmax()
    if peak_energy_slot == best_slot:
        recs.append(
            f"The {best_slot} is doubly optimal: it has both the highest completion rate "
            f"({_pct(best_rate)}) and the highest average energy "
            f"({tod_energy[best_slot]:.2f}/5) — align your hardest work here."
        )
    else:
        recs.append(
            f"Note: peak energy occurs in the {peak_energy_slot} "
            f"(avg {tod_energy[peak_energy_slot]:.2f}/5) but peak completion rate is in the "
            f"{best_slot} ({_pct(best_rate)}). Consider experimenting with task scheduling "
            f"to align both."
        )

    return recs


# ---------------------------------------------------------------------------
# 5. Full report
# ---------------------------------------------------------------------------

def full_recommendations_report(df: pd.DataFrame) -> dict:
    """
    Run all recommendation analyses and return a consolidated dict.

    Returns
    -------
    dict
        {
            "best_time_of_day":     { best_slot, completion_rates, lift_over_worst, session_counts },
            "best_task_by_energy":  { low, medium, high, all_rates },
            "distraction_threshold":{ threshold, performance_floor, rate_below/above, by_count },
            "recommendations":      [ "...", ... ],
        }
    """
    _validate(df)

    return {
        "best_time_of_day":      best_time_of_day(df),
        "best_task_by_energy":   best_task_by_energy(df),
        "distraction_threshold": distraction_threshold(df),
        "recommendations":       generate_recommendations(df),
    }
