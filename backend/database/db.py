"""
database/db.py
--------------
SQLite layer for the productivity analytics pipeline.

How it works
------------
1. The CSV is the seed source — raw data lives in data/productivity.csv.
2. On first run, init_db() creates productivity.db and loads the CSV into it.
3. All runtime data access goes through SQL queries, not the CSV directly.
   This mirrors real-world data pipelines and keeps analytics code
   independent of the storage format.

Public API
----------
Setup
  init_db()                          — create DB and seed from CSV (idempotent)
  query(sql, params)                 — run any SQL, return a DataFrame

Simple filters (return raw rows)
  get_all()                          — full dataset
  get_by_task_type(task_type)        — filter by task type
  get_by_time_of_day(time_of_day)    — filter by time of day

Aggregation queries (return summary stats)
  get_completion_rate_by_time_of_day()   — completion rate per time slot
  get_completion_rate_by_task_type()     — completion rate per task type
  get_avg_energy_by_completion()         — avg energy: completed vs not
  get_avg_distraction_by_completion()    — avg distractions: completed vs not
  get_completion_summary()               — full breakdown by task × time of day
  get_user_summary()                     — per-user stats

Deeper analytical queries
  get_high_distraction_sessions(threshold)  — sessions above distraction threshold
  get_top_performers(min_sessions, min_rate) — high-completion users
  get_session_duration_analysis()           — completion bucketed by session length
  get_focus_energy_by_task_time()           — focus/energy for every task × time combo
"""

from __future__ import annotations

import os
import sqlite3

import pandas as pd

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

_DB_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "productivity.db")
_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "productivity.csv")

# ---------------------------------------------------------------------------
# Table schema
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS productivity (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id              TEXT    NOT NULL,
    task_type            TEXT    NOT NULL,
    session_start        TEXT,
    session_end          TEXT,
    duration             INTEGER,
    time_of_day          TEXT,
    energy_level         INTEGER,
    focus_level          INTEGER,
    distraction_count    INTEGER,
    distraction_duration INTEGER,
    completed            INTEGER
);
"""

# ---------------------------------------------------------------------------
# Setup: connection, initialisation, core query runner
# ---------------------------------------------------------------------------

def _connect() -> sqlite3.Connection:
    """Open and return a connection to the SQLite database."""
    return sqlite3.connect(_DB_PATH)


def init_db() -> None:
    """
    Create the database and table if they don't exist, then seed from CSV.
    Safe to call on every startup — skips seeding if rows already exist.
    """
    if not os.path.exists(_CSV_PATH):
        raise FileNotFoundError(f"Seed CSV not found at {_CSV_PATH}")

    conn = _connect()
    try:
        cursor = conn.cursor()
        cursor.execute(_CREATE_TABLE_SQL)
        conn.commit()

        cursor.execute("SELECT COUNT(*) FROM productivity")
        count = cursor.fetchone()[0]

        if count == 0:
            df = pd.read_csv(_CSV_PATH)
            df.to_sql("productivity", conn, if_exists="append", index=False)
            conn.commit()
            print(f"[db] Seeded {len(df)} rows from CSV into productivity.db")
        else:
            print(f"[db] Database already contains {count} rows — skipping seed")
    finally:
        conn.close()


def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    """
    Run any parameterised SQL query and return the result as a DataFrame.
    Use ? as the placeholder for parameters.

    Example
    -------
    query("SELECT * FROM productivity WHERE task_type = ?", ("deep_work",))
    """
    conn = _connect()
    try:
        return pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Simple filters — return raw rows
# ---------------------------------------------------------------------------

def get_all() -> pd.DataFrame:
    """Return the full productivity dataset."""
    return query("SELECT * FROM productivity")


def get_by_task_type(task_type: str) -> pd.DataFrame:
    """Return all sessions for a specific task type (deep_work / creative / admin)."""
    return query(
        "SELECT * FROM productivity WHERE task_type = ?",
        (task_type,)
    )


def get_by_time_of_day(time_of_day: str) -> pd.DataFrame:
    """Return all sessions for a specific time-of-day slot (morning / afternoon / night)."""
    return query(
        "SELECT * FROM productivity WHERE time_of_day = ?",
        (time_of_day,)
    )


# ---------------------------------------------------------------------------
# Aggregation queries — completion rates
# ---------------------------------------------------------------------------

def get_completion_rate_by_time_of_day() -> pd.DataFrame:
    """
    Completion rate grouped by time_of_day.
    Returns: time_of_day, session_count, completion_rate — ordered best first.
    """
    return query("""
        SELECT
            time_of_day,
            COUNT(*)                 AS session_count,
            ROUND(AVG(completed), 4) AS completion_rate
        FROM productivity
        GROUP BY time_of_day
        ORDER BY completion_rate DESC
    """)


def get_completion_rate_by_task_type() -> pd.DataFrame:
    """
    Completion rate grouped by task_type.
    Returns: task_type, session_count, completion_rate — ordered best first.
    """
    return query("""
        SELECT
            task_type,
            COUNT(*)                 AS session_count,
            ROUND(AVG(completed), 4) AS completion_rate
        FROM productivity
        GROUP BY task_type
        ORDER BY completion_rate DESC
    """)


def get_avg_energy_by_completion() -> pd.DataFrame:
    """
    Average energy_level split by whether the session was completed (1) or not (0).
    Shows whether higher energy correlates with task completion.
    """
    return query("""
        SELECT
            completed,
            COUNT(*)                    AS session_count,
            ROUND(AVG(energy_level), 4) AS avg_energy_level
        FROM productivity
        GROUP BY completed
        ORDER BY completed DESC
    """)


def get_avg_distraction_by_completion() -> pd.DataFrame:
    """
    Average distraction_count split by whether the session was completed (1) or not (0).
    Shows whether fewer distractions correlates with task completion.
    """
    return query("""
        SELECT
            completed,
            COUNT(*)                         AS session_count,
            ROUND(AVG(distraction_count), 4) AS avg_distraction_count
        FROM productivity
        GROUP BY completed
        ORDER BY completed DESC
    """)


def get_completion_summary() -> pd.DataFrame:
    """
    Full breakdown of completion rate, energy, focus, and distractions
    grouped by every task_type × time_of_day combination.
    Useful for spotting the best and worst scheduling combinations.
    """
    return query("""
        SELECT
            task_type,
            time_of_day,
            COUNT(*)                         AS session_count,
            ROUND(AVG(completed), 4)         AS completion_rate,
            ROUND(AVG(energy_level), 2)      AS avg_energy,
            ROUND(AVG(focus_level), 2)       AS avg_focus,
            ROUND(AVG(distraction_count), 2) AS avg_distractions,
            ROUND(AVG(duration), 1)          AS avg_duration
        FROM productivity
        GROUP BY task_type, time_of_day
        ORDER BY completion_rate DESC
    """)


def get_user_summary() -> pd.DataFrame:
    """
    Per-user stats: session count, completion rate, avg energy, focus, distractions.
    Ordered by completion rate descending — top performers appear first.
    """
    return query("""
        SELECT
            user_id,
            COUNT(*)                         AS total_sessions,
            ROUND(AVG(completed), 4)         AS completion_rate,
            ROUND(AVG(energy_level), 2)      AS avg_energy,
            ROUND(AVG(focus_level), 2)       AS avg_focus,
            ROUND(AVG(distraction_count), 2) AS avg_distractions,
            ROUND(AVG(duration), 1)          AS avg_duration
        FROM productivity
        GROUP BY user_id
        ORDER BY completion_rate DESC
    """)


# ---------------------------------------------------------------------------
# Deeper analytical queries
# ---------------------------------------------------------------------------

def get_high_distraction_sessions(threshold: int = 8) -> pd.DataFrame:
    """
    Return sessions where distraction_count is at or above the threshold.
    Default threshold is 8, which is the point where performance drops sharply.
    Ordered by distraction count descending.
    """
    return query(
        """
        SELECT
            user_id, task_type, time_of_day,
            energy_level, focus_level,
            distraction_count, duration, completed
        FROM productivity
        WHERE distraction_count >= ?
        ORDER BY distraction_count DESC
        """,
        (threshold,)
    )


def get_top_performers(min_sessions: int = 5, min_completion_rate: float = 0.6) -> pd.DataFrame:
    """
    Users who meet both a minimum session count and a minimum completion rate.
    Filters out users with too few sessions (statistically unreliable).
    Ordered by completion rate descending.
    """
    return query(
        """
        SELECT
            user_id,
            COUNT(*)                         AS total_sessions,
            ROUND(AVG(completed), 4)         AS completion_rate,
            ROUND(AVG(energy_level), 2)      AS avg_energy,
            ROUND(AVG(focus_level), 2)       AS avg_focus,
            ROUND(AVG(distraction_count), 2) AS avg_distractions,
            ROUND(AVG(duration), 1)          AS avg_duration
        FROM productivity
        GROUP BY user_id
        HAVING total_sessions >= ? AND completion_rate >= ?
        ORDER BY completion_rate DESC
        """,
        (min_sessions, min_completion_rate)
    )


def get_session_duration_analysis() -> pd.DataFrame:
    """
    Completion rate, focus, and energy bucketed by session length:
    short (<30 min), medium (30–60), long (60–90), very long (90+).
    Useful for understanding whether longer sessions are more or less productive.
    """
    return query("""
        SELECT
            CASE
                WHEN duration < 30  THEN 'short (<30 min)'
                WHEN duration < 60  THEN 'medium (30-60 min)'
                WHEN duration < 90  THEN 'long (60-90 min)'
                ELSE                     'very long (90+ min)'
            END                          AS duration_band,
            COUNT(*)                     AS session_count,
            ROUND(AVG(completed), 4)     AS completion_rate,
            ROUND(AVG(focus_level), 2)   AS avg_focus,
            ROUND(AVG(energy_level), 2)  AS avg_energy,
            ROUND(AVG(distraction_count), 2) AS avg_distractions
        FROM productivity
        GROUP BY duration_band
        ORDER BY MIN(duration)
    """)


def get_focus_energy_by_task_time() -> pd.DataFrame:
    """
    Average focus, energy, and completion rate for every task_type × time_of_day
    combination. Useful for identifying the optimal scheduling windows per task type,
    and as the data source for a heatmap visualisation.
    """
    return query("""
        SELECT
            task_type,
            time_of_day,
            COUNT(*)                        AS session_count,
            ROUND(AVG(focus_level), 2)      AS avg_focus,
            ROUND(AVG(energy_level), 2)     AS avg_energy,
            ROUND(AVG(completed), 4)        AS completion_rate
        FROM productivity
        GROUP BY task_type, time_of_day
        ORDER BY task_type, time_of_day
    """)
