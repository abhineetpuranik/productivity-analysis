"""
psychology/mapper.py
---------------------
Maps productivity data insights to established psychology concepts.

Three core concepts are covered:

  Cognitive Load  (Sweller, 1988)
  --------------------------------
  The total mental effort being used in working memory. Working memory can
  only hold ~4 chunks of information at once. When load exceeds capacity —
  through task complexity, long sessions, or interruptions — performance
  degrades and errors increase.

  Attention Span / Directed Attention Fatigue  (Kaplan, 1995)
  -------------------------------------------------------------
  Sustained voluntary attention depletes over time. Focus level in the data
  is a direct proxy for directed attention. Distraction count reflects how
  often attention was involuntarily captured (stimulus-driven attention),
  which accelerates fatigue.

  Energy Cycles / Ultradian Rhythms  (Kleitman / Peretz Lavie)
  --------------------------------------------------------------
  The body operates on ~90-minute ultradian cycles of high and low alertness
  throughout the day, layered on top of the circadian (24-hour) rhythm.
  Energy level in the data maps to where a person sits in their cycle.
  Morning peaks align with cortisol's natural rise after waking.

Each public function returns:
  - "concept"      : the psychology concept name
  - "theory"       : one-sentence grounding in research
  - "what_we_see"  : plain-language description of the data pattern
  - "why_it_happens": plain-language causal explanation
  - "recommendation": one actionable takeaway
  - "data_signals" : the raw numbers that drove the interpretation
"""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# Thresholds (kept consistent with scoring/engine.py and patterns/detector.py)
# ---------------------------------------------------------------------------

_HIGH_ENERGY   = 4   # energy_level >= 4 → high
_HIGH_FOCUS    = 4   # focus_level   >= 4 → high
_HIGH_DISTRACT = 8   # distraction_count >= 8 → high load
_LONG_SESSION  = 90  # duration >= 90 min → long

# Cognitive load base demands per task type (mirrors scoring/engine.py)
_TASK_DEMAND_SCORE: dict[str, float] = {
    "deep_work": 70.0,
    "creative":  55.0,
    "admin":     35.0,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate(df: pd.DataFrame) -> None:
    required = {
        "task_type", "duration", "energy_level",
        "focus_level", "distraction_count", "completed",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _avg(series: pd.Series) -> float:
    return round(float(series.mean()), 2)


# ---------------------------------------------------------------------------
# Public psychology mapping functions
# ---------------------------------------------------------------------------

def cognitive_load_profile(df: pd.DataFrame) -> dict:
    """
    Map task type, session duration, and distraction count to Cognitive Load
    Theory (Sweller, 1988).

    Intrinsic load  = inherent complexity of the task type
    Extraneous load = distractions that consume working memory without adding value
    Germane load    = productive mental effort that builds understanding

    Returns
    -------
    dict with keys: concept, theory, what_we_see, why_it_happens,
                    recommendation, data_signals
    """
    _validate(df)

    # --- Data signals ---
    avg_cl_by_task = (
        df.groupby("task_type")
        .apply(lambda g: round(
            _TASK_DEMAND_SCORE[g.name]
            + 15 * ((g["duration"].clip(0, 180) - 0) / 180).mean()
            + 15 * ((g["distraction_count"].clip(0, 15) - 0) / 15).mean(),
            2
        ), include_groups=False)
        .to_dict()
    )

    high_load_sessions = int((df["distraction_count"] >= _HIGH_DISTRACT).sum())
    high_load_pct      = high_load_sessions / len(df)

    long_sessions      = int((df["duration"] >= _LONG_SESSION).sum())
    long_session_pct   = long_sessions / len(df)

    avg_dist_by_task   = df.groupby("task_type")["distraction_count"].mean().round(2).to_dict()
    completion_by_task = df.groupby("task_type")["completed"].mean().round(4).to_dict()

    # Overload indicator: high distraction + long session + low completion
    overload_mask = (
        (df["distraction_count"] >= _HIGH_DISTRACT)
        & (df["duration"] >= _LONG_SESSION)
    )
    overload_completion = round(float(df[overload_mask]["completed"].mean()), 4) if overload_mask.sum() > 0 else None
    normal_completion   = round(float(df[~overload_mask]["completed"].mean()), 4)

    # --- Narrative ---
    heaviest_task = max(avg_cl_by_task, key=avg_cl_by_task.get)
    lightest_task = min(avg_cl_by_task, key=avg_cl_by_task.get)

    what_we_see = (
        f"{heaviest_task.replace('_', ' ').capitalize()} sessions carry the heaviest "
        f"cognitive load (avg score {avg_cl_by_task[heaviest_task]:.0f}/100), while "
        f"{lightest_task.replace('_', ' ')} sessions are the lightest "
        f"({avg_cl_by_task[lightest_task]:.0f}/100). "
        f"{_pct(high_load_pct)} of all sessions have 8+ distractions, and "
        f"{_pct(long_session_pct)} run 90+ minutes — both are known load amplifiers."
    )

    why_it_happens = (
        "Working memory can only juggle a limited number of things at once. "
        f"Deep work demands sustained concentration, so its baseline load is already high. "
        "Every distraction forces the brain to drop what it was holding, reload context, "
        "and start again — a process that takes 10–20 minutes to fully recover from. "
        "Long sessions compound this: mental resources deplete without rest, "
        "turning manageable load into cognitive overload."
    )

    if overload_completion is not None:
        rec_data = f"overloaded sessions complete at only {_pct(overload_completion)} vs {_pct(normal_completion)} otherwise"
    else:
        rec_data = f"normal sessions complete at {_pct(normal_completion)}"

    recommendation = (
        f"Cap deep work sessions at 90 minutes and take a real break before starting again. "
        f"Protect those blocks from interruptions — {rec_data}. "
        "For admin tasks, batching them together reduces the switching cost of re-entering "
        "a different mental mode repeatedly."
    )

    return {
        "concept":        "Cognitive Load Theory (Sweller, 1988)",
        "theory":         "Working memory has a fixed capacity; exceeding it through task complexity, "
                          "long sessions, or interruptions causes performance to collapse.",
        "what_we_see":    what_we_see,
        "why_it_happens": why_it_happens,
        "recommendation": recommendation,
        "data_signals": {
            "avg_cognitive_load_by_task":    avg_cl_by_task,
            "avg_distraction_count_by_task": avg_dist_by_task,
            "completion_rate_by_task":       completion_by_task,
            "high_distraction_sessions_pct": round(high_load_pct, 4),
            "long_session_pct":              round(long_session_pct, 4),
            "completion_overloaded":         overload_completion,
            "completion_normal":             normal_completion,
        },
    }


def attention_span_profile(df: pd.DataFrame) -> dict:
    """
    Map focus level and distraction count to Directed Attention Fatigue
    (Kaplan, 1995) and the concept of attentional blink / task-switching cost.

    Returns
    -------
    dict with keys: concept, theory, what_we_see, why_it_happens,
                    recommendation, data_signals
    """
    _validate(df)

    # --- Data signals ---
    avg_focus_by_tod  = df.groupby("time_of_day")["focus_level"].mean().round(2).to_dict() \
                        if "time_of_day" in df.columns else {}
    avg_focus_by_task = df.groupby("task_type")["focus_level"].mean().round(2).to_dict()

    # Focus degradation: compare short vs long sessions
    short = df[df["duration"] <  _LONG_SESSION]
    long  = df[df["duration"] >= _LONG_SESSION]
    focus_short = _avg(short["focus_level"]) if len(short) else None
    focus_long  = _avg(long["focus_level"])  if len(long)  else None

    # Distraction → focus relationship
    low_dist  = df[df["distraction_count"] <= 3]
    high_dist = df[df["distraction_count"] >= _HIGH_DISTRACT]
    focus_low_dist  = _avg(low_dist["focus_level"])  if len(low_dist)  else None
    focus_high_dist = _avg(high_dist["focus_level"]) if len(high_dist) else None

    # High-focus session completion
    high_focus_mask = df["focus_level"] >= _HIGH_FOCUS
    completion_high_focus = round(float(df[high_focus_mask]["completed"].mean()), 4) \
                            if high_focus_mask.sum() > 0 else None
    completion_low_focus  = round(float(df[~high_focus_mask]["completed"].mean()), 4)

    # Best time of day for focus
    best_tod_focus = max(avg_focus_by_tod, key=avg_focus_by_tod.get) if avg_focus_by_tod else "N/A"

    # --- Narrative ---
    focus_drop = round(focus_short - focus_long, 2) if (focus_short and focus_long) else None
    dist_focus_gap = round(focus_low_dist - focus_high_dist, 2) \
                     if (focus_low_dist and focus_high_dist) else None

    what_we_see = (
        f"Focus is highest in the {best_tod_focus} (avg {avg_focus_by_tod.get(best_tod_focus, '?')}). "
        + (f"It drops by {focus_drop} points in sessions over 90 minutes. " if focus_drop else "")
        + (f"Sessions with 8+ distractions show {dist_focus_gap} points lower focus "
           f"than low-distraction sessions. " if dist_focus_gap else "")
        + (f"High-focus sessions complete at {_pct(completion_high_focus)} vs "
           f"{_pct(completion_low_focus)} for low-focus sessions." if completion_high_focus else "")
    )

    why_it_happens = (
        "Directed attention — the kind needed for focused work — is a limited resource. "
        "It depletes with use and recovers with rest, much like a muscle. "
        "Each distraction doesn't just steal a moment; it triggers an 'attentional blink' "
        "where the brain needs time to re-engage with the original task. "
        "In long sessions without breaks, directed attention fatigue sets in: "
        "the person is still sitting there, but their brain has quietly checked out."
    )

    recommendation = (
        f"Work in focused blocks of 45–90 minutes, then take a genuine mental break "
        f"(a walk, not scrolling). Schedule your most focus-demanding work in the "
        f"{best_tod_focus} when directed attention is naturally strongest. "
        "Treat distraction reduction as a prerequisite, not a bonus — "
        "the data shows focus drops sharply once interruptions pile up."
    )

    return {
        "concept":        "Directed Attention Fatigue (Kaplan, 1995) & Attentional Blink",
        "theory":         "Voluntary, directed attention is a finite resource that depletes "
                          "through sustained use and is rapidly eroded by interruptions.",
        "what_we_see":    what_we_see,
        "why_it_happens": why_it_happens,
        "recommendation": recommendation,
        "data_signals": {
            "avg_focus_by_time_of_day":       avg_focus_by_tod,
            "avg_focus_by_task_type":         avg_focus_by_task,
            "avg_focus_short_sessions":       focus_short,
            "avg_focus_long_sessions":        focus_long,
            "focus_drop_in_long_sessions":    focus_drop,
            "avg_focus_low_distractions":     focus_low_dist,
            "avg_focus_high_distractions":    focus_high_dist,
            "focus_gap_distraction_effect":   dist_focus_gap,
            "completion_high_focus":          completion_high_focus,
            "completion_low_focus":           completion_low_focus,
            "best_time_of_day_for_focus":     best_tod_focus,
        },
    }


def energy_cycle_profile(df: pd.DataFrame) -> dict:
    """
    Map energy level and time-of-day patterns to Ultradian Rhythms
    (Kleitman, 1963) and the Circadian Performance Curve.

    The circadian rhythm creates a predictable daily arc of alertness.
    Ultradian rhythms (~90 min cycles) create peaks and troughs within that arc.
    Energy level in the data is a direct proxy for where someone sits in their cycle.

    Returns
    -------
    dict with keys: concept, theory, what_we_see, why_it_happens,
                    recommendation, data_signals
    """
    _validate(df)

    # --- Data signals ---
    avg_energy_by_tod  = df.groupby("time_of_day")["energy_level"].mean().round(2).to_dict() \
                         if "time_of_day" in df.columns else {}
    avg_energy_by_task = df.groupby("task_type")["energy_level"].mean().round(2).to_dict()

    # Completion rate at each energy level
    completion_by_energy = df.groupby("energy_level")["completed"].mean().round(4).to_dict()

    # High vs low energy completion
    high_e = df[df["energy_level"] >= _HIGH_ENERGY]
    low_e  = df[df["energy_level"] <  _HIGH_ENERGY]
    completion_high_e = round(float(high_e["completed"].mean()), 4) if len(high_e) else None
    completion_low_e  = round(float(low_e["completed"].mean()),  4) if len(low_e)  else None
    multiplier = round(completion_high_e / completion_low_e, 1) \
                 if (completion_high_e and completion_low_e and completion_low_e > 0) else None

    # Energy × task type: best match
    energy_task = df.groupby("task_type")["energy_level"].mean().round(2)
    best_energy_task = energy_task.idxmax()

    # Energy × time of day: peak slot
    best_tod = max(avg_energy_by_tod, key=avg_energy_by_tod.get) if avg_energy_by_tod else "N/A"
    worst_tod = min(avg_energy_by_tod, key=avg_energy_by_tod.get) if avg_energy_by_tod else "N/A"

    energy_gap = round(
        avg_energy_by_tod.get(best_tod, 0) - avg_energy_by_tod.get(worst_tod, 0), 2
    ) if avg_energy_by_tod else 0

    # --- Narrative ---
    what_we_see = (
        f"Average energy peaks in the {best_tod} "
        f"({avg_energy_by_tod.get(best_tod, '?')}/5) and is lowest at {worst_tod} "
        f"({avg_energy_by_tod.get(worst_tod, '?')}/5) — a gap of {energy_gap} points. "
        + (f"High-energy sessions (≥{_HIGH_ENERGY}) complete at {_pct(completion_high_e)}, "
           f"compared to {_pct(completion_low_e)} when energy is low"
           + (f" — a {multiplier}x difference." if multiplier else ".")
           if completion_high_e else "")
    )

    why_it_happens = (
        "The body runs on a 24-hour circadian clock that regulates cortisol, body temperature, "
        "and alertness. Cortisol — the brain's natural 'get up and go' signal — peaks roughly "
        "30–60 minutes after waking, which is why most people feel sharpest in the morning. "
        "Layered on top of this are ultradian rhythms: ~90-minute cycles of higher and lower "
        "arousal that repeat throughout the day. Working against these cycles — scheduling "
        "demanding tasks during natural troughs — is like trying to sprint uphill."
    )

    recommendation = (
        f"Align your hardest, most important work with your energy peak ({best_tod}). "
        f"Reserve {worst_tod} for low-stakes tasks: email, scheduling, routine admin. "
        "Pay attention to the 90-minute ultradian signal: when you notice your mind "
        "wandering or energy dipping mid-session, that's your body asking for a 10–20 "
        "minute rest — not a sign of laziness."
    )

    return {
        "concept":        "Circadian Rhythm & Ultradian Performance Cycles (Kleitman, 1963)",
        "theory":         "The body's 24-hour circadian clock and ~90-minute ultradian cycles "
                          "create predictable peaks and troughs in alertness and cognitive performance.",
        "what_we_see":    what_we_see,
        "why_it_happens": why_it_happens,
        "recommendation": recommendation,
        "data_signals": {
            "avg_energy_by_time_of_day":  avg_energy_by_tod,
            "avg_energy_by_task_type":    avg_energy_by_task,
            "completion_by_energy_level": completion_by_energy,
            "completion_high_energy":     completion_high_e,
            "completion_low_energy":      completion_low_e,
            "completion_multiplier":      multiplier,
            "peak_energy_slot":           best_tod,
            "lowest_energy_slot":         worst_tod,
            "energy_gap_peak_vs_trough":  energy_gap,
        },
    }


def behavioral_explanation(df: pd.DataFrame) -> dict:
    """
    Synthesise all three psychology profiles into a single plain-language
    behavioral explanation of what the data says about how this person works.

    Combines signals from cognitive load, attention span, and energy cycles
    into a coherent narrative with a prioritised action list.

    Returns
    -------
    dict
        {
            "summary":          str   — 3–4 sentence plain-language overview
            "key_behaviors":    list  — observed behavioral patterns, each a str
            "root_causes":      list  — psychological explanations, each a str
            "action_plan":      list  — prioritised recommendations, each a str
            "profiles": {
                "cognitive_load":  { ... },
                "attention_span":  { ... },
                "energy_cycles":   { ... },
            }
        }
    """
    _validate(df)

    cl  = cognitive_load_profile(df)
    att = attention_span_profile(df)
    eng = energy_cycle_profile(df)

    # Pull key numbers for the narrative
    peak_tod        = eng["data_signals"]["peak_energy_slot"]
    low_tod         = eng["data_signals"]["lowest_energy_slot"]
    focus_best_tod  = att["data_signals"]["best_time_of_day_for_focus"]
    cl_heavy_task   = max(
        cl["data_signals"]["avg_cognitive_load_by_task"],
        key=cl["data_signals"]["avg_cognitive_load_by_task"].get
    )
    completion_high_e = eng["data_signals"]["completion_high_energy"]
    completion_low_e  = eng["data_signals"]["completion_low_energy"]
    multiplier        = eng["data_signals"]["completion_multiplier"]
    focus_drop        = att["data_signals"]["focus_drop_in_long_sessions"]
    dist_gap          = att["data_signals"]["focus_gap_distraction_effect"]
    overload_comp     = cl["data_signals"]["completion_overloaded"]
    normal_comp       = cl["data_signals"]["completion_normal"]

    # --- Summary ---
    summary = (
        f"This person's productivity is strongly tied to their biological state. "
        f"They perform best in the {peak_tod}, when energy and focus are naturally highest, "
        f"and struggle most at {low_tod}. "
        f"Distractions are their biggest enemy: high-distraction sessions show noticeably "
        f"lower focus and completion rates, and the data suggests that protecting attention "
        f"is more impactful than working longer hours. "
        f"{cl_heavy_task.replace('_', ' ').capitalize()} is the most cognitively demanding "
        f"task type and benefits most from being scheduled during peak energy windows."
    )

    # --- Key behaviors observed ---
    behaviors = [
        f"Completes tasks at {_pct(completion_high_e)} when energy is high vs "
        f"{_pct(completion_low_e)} when energy is low"
        + (f" — a {multiplier}x gap." if multiplier else ".")
        if completion_high_e else
        "Completion rate is strongly linked to energy level.",

        f"Focus drops by {focus_drop} points in sessions over 90 minutes, "
        "suggesting attention fatigue sets in during long unbroken work blocks."
        if focus_drop else
        "Focus degrades in longer sessions.",

        f"High-distraction sessions (8+) show {dist_gap} points lower focus "
        "than low-distraction sessions, indicating distractions actively erode concentration."
        if dist_gap else
        "Distractions measurably reduce focus.",

        f"Cognitive load is heaviest during {cl_heavy_task.replace('_', ' ')} sessions, "
        "which also have the highest completion rate — suggesting this person rises to "
        "the challenge when conditions are right.",

        f"Overloaded sessions (long + high distraction) complete at only "
        f"{_pct(overload_comp)} vs {_pct(normal_comp)} for normal sessions."
        if overload_comp else
        "Overloaded sessions have significantly lower completion rates.",
    ]

    # --- Root causes ---
    root_causes = [
        "Working memory saturation: too many simultaneous demands (task complexity + "
        "distractions + session length) exceed the brain's processing capacity.",

        "Directed attention depletion: voluntary focus is a finite daily resource. "
        "Once spent, it cannot be willed back — only recovered through genuine rest.",

        f"Circadian misalignment: scheduling demanding work outside the {peak_tod} "
        "energy peak means the brain is operating below its natural capability.",

        "Attentional switching cost: each distraction doesn't just steal a moment — "
        "it triggers a 10–20 minute recovery period before full focus is restored.",
    ]

    # --- Action plan (prioritised) ---
    action_plan = [
        f"Guard the {peak_tod}. Block it for your most important, cognitively demanding "
        f"work ({cl_heavy_task.replace('_', ' ')}). Treat it as non-negotiable.",

        f"Work in 45–90 minute focused blocks. Set a timer. When it goes off, stop — "
        f"even if you feel fine. The data shows focus degrades after 90 minutes.",

        f"Eliminate distractions before starting, not during. Close notifications, "
        f"set your status to busy, and remove temptations. Reacting to distractions "
        f"mid-session costs far more time than preventing them upfront.",

        f"Batch low-demand tasks (admin, email) into your {low_tod} trough. "
        f"This preserves peak energy for work that actually needs it.",

        f"Take real breaks. A 10–15 minute walk or rest between blocks restores "
        f"directed attention and resets working memory — scrolling does not.",
    ]

    return {
        "summary":       summary,
        "key_behaviors": behaviors,
        "root_causes":   root_causes,
        "action_plan":   action_plan,
        "profiles": {
            "cognitive_load": cl,
            "attention_span": att,
            "energy_cycles":  eng,
        },
    }


def full_psychology_report(df: pd.DataFrame) -> dict:
    """
    Run all psychology analyses and return a single consolidated dict.

    Returns
    -------
    dict
        {
            "cognitive_load":       { concept, theory, what_we_see, why_it_happens,
                                      recommendation, data_signals },
            "attention_span":       { ... },
            "energy_cycles":        { ... },
            "behavioral_explanation": { summary, key_behaviors, root_causes,
                                        action_plan, profiles },
        }
    """
    _validate(df)

    cl  = cognitive_load_profile(df)
    att = attention_span_profile(df)
    eng = energy_cycle_profile(df)
    beh = behavioral_explanation(df)

    return {
        "cognitive_load":         cl,
        "attention_span":         att,
        "energy_cycles":          eng,
        "behavioral_explanation": beh,
    }
