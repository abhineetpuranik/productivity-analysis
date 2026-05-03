"""
app.py
------
Flask API entry point for the BPAE productivity analytics pipeline.

Endpoints
---------
GET /analyze  — Runs the full analytics pipeline and returns a structured JSON response.
GET /groq     — Sends the AI prompt to the Groq LLM and returns a coaching insight.
GET /health   — Simple liveness check.

Pipeline order (inside /analyze)
---------------------------------
1. Load data from SQLite
2. Run analytics, pattern detection, scoring, psychology, segmentation,
   data-driven recommendations, and chart generation
3. Assemble and return a single JSON response
"""

from __future__ import annotations

import os
import traceback

from dotenv import load_dotenv
load_dotenv()  # must run before any os.environ access

from flask import Flask, jsonify
from flask_cors import CORS
from openai import OpenAI

from analytics.productivity import full_report as analytics_report
from analytics.recommendations import full_recommendations_report
from analytics.segmentation import full_segmentation_report
from analytics.visualizations import generate_all_charts
from database.db import get_all, init_db
from patterns.detector import all_patterns
from psychology.mapper import full_psychology_report
from scoring.engine import score_dataset

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)  # allow all origins — tighten with origins=[...] in production

# Initialise SQLite database on startup (idempotent — safe to call every run)
init_db()

# Cache the AI prompt after /analyze runs so /grok doesn't recompute everything
_cached_prompt: str | None = None


# ---------------------------------------------------------------------------
# Helper: data loading
# ---------------------------------------------------------------------------

def _load_data():
    """Load the full productivity dataset from SQLite."""
    return get_all()


# ---------------------------------------------------------------------------
# Helpers: response assembly
# ---------------------------------------------------------------------------

def _build_metrics(analytics: dict, scoring: dict) -> dict:
    """
    Combine analytics and scoring results into the top-level metrics block.
    The scored DataFrame is excluded — it is not JSON-serialisable as-is.
    """
    return {
        "completion_rate": {
            "by_time_of_day": analytics["completion_rate_by_time_of_day"],
            "by_task_type":   analytics["completion_rate_by_task_type"],
        },
        "correlations":        analytics["correlations"],
        "productivity_score":  analytics["productivity_score"],
        "productivity_index": {
            "summary":        scoring["pi_summary"],
            "by_task_type":   scoring["by_task_type"],
            "by_time_of_day": scoring["by_time_of_day"],
        },
        "cognitive_load_score": {
            "summary": scoring["cl_summary"],
        },
        "score_weights": scoring["score_weights"],
    }


def _build_insights(patterns: dict, psychology: dict) -> list[str]:
    """
    Collect all human-readable insight strings into a flat list.
    Order: pattern insights first, then psychology observations.
    """
    insights: list[str] = []

    for value in patterns.values():
        if isinstance(value, dict) and "insight" in value:
            insights.append(value["insight"])

    for concept in ("cognitive_load", "attention_span", "energy_cycles"):
        profile = psychology.get(concept, {})
        if "what_we_see" in profile:
            insights.append(profile["what_we_see"])

    return insights


def _build_recommendations(psychology: dict) -> list[str]:
    """
    Collect one recommendation per psychology concept, then append the
    prioritised action plan steps from the behavioral explanation.
    """
    recommendations: list[str] = []

    for concept in ("cognitive_load", "attention_span", "energy_cycles"):
        profile = psychology.get(concept, {})
        if "recommendation" in profile:
            recommendations.append(profile["recommendation"])

    action_plan = psychology.get("behavioral_explanation", {}).get("action_plan", [])
    recommendations.extend(action_plan)

    return recommendations


def _build_ai_prompt(metrics: dict, insights: list[str], beh: dict) -> str:
    """
    Compose a ready-to-paste prompt for any LLM.
    Summarises the key findings and asks for 3 evidence-based recommendations.
    """
    completion_tod  = metrics["completion_rate"]["by_time_of_day"]
    completion_task = metrics["completion_rate"]["by_task_type"]
    pi_mean         = metrics["productivity_index"]["summary"]["mean"]
    cl_mean         = metrics["cognitive_load_score"]["summary"]["mean"]
    summary         = beh.get("summary", "")

    best_tod  = max(completion_tod,  key=completion_tod.get)
    best_task = max(completion_task, key=completion_task.get)

    insight_block = "\n".join(f"- {i}" for i in insights[:5])

    return (
        f"You are a productivity coach analysing a user's work session data.\n\n"
        f"Key findings:\n"
        f"- Best time of day: {best_tod} ({completion_tod[best_tod]*100:.1f}% completion rate)\n"
        f"- Best task type: {best_task.replace('_', ' ')} "
        f"({completion_task[best_task]*100:.1f}% completion rate)\n"
        f"- Average Productivity Index: {pi_mean}/100\n"
        f"- Average Cognitive Load Score: {cl_mean}/100\n\n"
        f"Data insights:\n{insight_block}\n\n"
        f"Behavioral summary:\n{summary}\n\n"
        f"Based on this data, provide 3 specific, evidence-based recommendations "
        f"to help this person improve their productivity. Be concise and practical."
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/analyze", methods=["GET"])
def analyze():
    """
    Run the full analytics pipeline and return a structured JSON response.

    Response keys
    -------------
    metrics                — completion rates, correlations, scores
    insights               — flat list of human-readable findings
    recommendations        — psychology-based action items
    data_recommendations   — purely data-computed recommendations
    behavioral_explanation — summary, key behaviors, root causes, action plan
    segmentation           — user segments, comparison, significance tests
    charts                 — base64-encoded PNG visualizations
    ai_prompt              — ready-to-paste LLM prompt
    """
    try:
        df = _load_data()

        # Run all pipeline modules
        analytics    = analytics_report(df)
        patterns     = all_patterns(df)
        scoring      = score_dataset(df)
        psychology   = full_psychology_report(df)
        segmentation = full_segmentation_report(df)
        data_recs    = full_recommendations_report(df)
        charts       = generate_all_charts(df)

        # Assemble response
        metrics  = _build_metrics(analytics, scoring)
        insights = _build_insights(patterns, psychology)
        recs     = _build_recommendations(psychology)
        beh      = psychology["behavioral_explanation"]
        beh_out  = {
            "summary":       beh["summary"],
            "key_behaviors": beh["key_behaviors"],
            "root_causes":   beh["root_causes"],
            "action_plan":   beh["action_plan"],
        }
        ai_prompt = _build_ai_prompt(metrics, insights, beh)

        # Cache the prompt so /grok can use it without rerunning the pipeline
        global _cached_prompt
        _cached_prompt = ai_prompt

        return jsonify({
            "metrics":                metrics,
            "insights":               insights,
            "recommendations":        recs,
            "data_recommendations":   data_recs,
            "behavioral_explanation": beh_out,
            "segmentation":           segmentation,
            "charts":                 charts,
            "ai_prompt":              ai_prompt,
        })

    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404

    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@app.route("/groq", methods=["GET"])
def grok_insight():
    """
    Send the cached AI prompt to Groq (groq.com) and return the model's response.
    The prompt is built by /analyze — call that first if the cache is empty.

    Response: { "insight": "..." } on success, { "error": "..." } on failure.
    """
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        return jsonify({"error": "GROQ_API_KEY is not set. Add it to your .env file."}), 500

    try:
        # Use cached prompt if available, otherwise build it fresh
        global _cached_prompt
        if _cached_prompt:
            prompt = _cached_prompt
        else:
            df         = _load_data()
            analytics  = analytics_report(df)
            patterns   = all_patterns(df)
            scoring    = score_dataset(df)
            psychology = full_psychology_report(df)
            metrics    = _build_metrics(analytics, scoring)
            insights   = _build_insights(patterns, psychology)
            beh        = psychology["behavioral_explanation"]
            prompt     = _build_ai_prompt(metrics, insights, beh)
            _cached_prompt = prompt

        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
        )

        return jsonify({"insight": response.choices[0].message.content})

    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404

    except Exception as exc:
        return jsonify({"error": str(exc).split("\n")[0]}), 500


@app.route("/health", methods=["GET"])
def health():
    """Simple liveness check."""
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5000)
