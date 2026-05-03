# BPAE Dashboard

**Behavioral · Psychological · AI · Engagement**

A productivity analytics platform that takes raw work session data and turns it into evidence-based insights. It combines SQL data pipelines, statistical analysis, psychology theory, user segmentation, and AI-generated coaching into a single dashboard.

---

## What it does

You give it a dataset of work sessions — each row is one session with fields like task type, energy level, focus level, distraction count, and whether the task was completed. The platform runs it through a full analytics pipeline and surfaces:

- Completion rates broken down by time of day and task type
- Correlation analysis between energy, focus, distractions, and outcomes
- Productivity Index and Cognitive Load scores per session
- Pattern detection (best time of day, flow state, distraction impact)
- User segmentation into High / Mid / Low Performers with statistical significance testing
- Data-driven recommendations computed entirely from the numbers
- Psychology-grounded behavioral explanations (Cognitive Load Theory, Directed Attention Fatigue, Ultradian Rhythms)
- Two matplotlib visualizations rendered directly in the browser
- On-demand AI coaching via the Groq LLM API

---

## Project structure

```
bpae-dashboard/
├── backend/                  # Python / Flask API
│   ├── app.py                # Entry point — Flask routes
│   ├── data/
│   │   ├── productivity.csv  # Seed data (400 work sessions)
│   │   └── productivity.db   # SQLite database (auto-created on first run)
│   ├── database/
│   │   └── db.py             # SQL layer — init, queries, aggregations
│   ├── analytics/
│   │   ├── productivity.py   # Core metrics and correlations
│   │   ├── segmentation.py   # User segmentation + t-tests
│   │   ├── recommendations.py# Data-driven recommendations
│   │   └── visualizations.py # Matplotlib charts → base64 PNG
│   ├── scoring/
│   │   └── engine.py         # Productivity Index + Cognitive Load scores
│   ├── patterns/
│   │   └── detector.py       # Pattern detection (flow state, distraction impact, etc.)
│   ├── psychology/
│   │   └── mapper.py         # Maps data to psychology concepts
│   ├── .env                  # API keys (not committed)
│   └── requirements.txt      # Python dependencies
│
└── frontend/
    └── bpae-dashboard/       # Angular 21 app
        └── src/app/
            ├── components/   # One folder per UI section
            ├── services/     # HTTP client + caching
            └── models/       # TypeScript interfaces matching the API
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend API | Python 3, Flask |
| Database | SQLite (via Python's built-in `sqlite3`) |
| Data processing | Pandas, NumPy |
| Statistics | SciPy (`ttest_ind`) |
| Visualizations | Matplotlib |
| AI / LLM | Groq API (`llama-3.3-70b-versatile`) |
| Frontend | Angular 21, TypeScript |
| Styling | SCSS with CSS custom properties |

---

## Getting started

### Backend

```bash
cd backend
pip install -r requirements.txt
python app.py
```

The server starts on `http://localhost:5000`. On first run it automatically creates `productivity.db` and seeds it from the CSV.

Create a `.env` file in the `backend/` folder:

```
GROQ_API_KEY=your_key_here
```

You can get a free API key at [console.groq.com](https://console.groq.com).

### Frontend

```bash
cd frontend/bpae-dashboard
npm install
npm start
```

The app opens at `http://localhost:4200`. It calls the backend at `localhost:5000` automatically.

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/analyze` | Runs the full pipeline and returns all analytics data |
| GET | `/grok` | Sends the AI prompt to Groq and returns a coaching insight |
| GET | `/health` | Simple liveness check |

---

## Backend files explained

### `app.py`
The Flask entry point. Defines three routes (`/analyze`, `/grok`, `/health`), initialises the database on startup, and assembles the final JSON response from all the pipeline modules. If you want to understand the full data flow, start here.

---

### `database/db.py`
The SQL layer. All data access goes through this file — nothing reads the CSV directly at runtime.

- `init_db()` — creates `productivity.db` and seeds it from the CSV on first run. Safe to call every startup (skips if data already exists).
- `query(sql, params)` — the core runner. Takes any SQL string and returns a pandas DataFrame.
- Simple filter functions: `get_all()`, `get_by_task_type()`, `get_by_time_of_day()`
- Aggregation queries: `get_completion_rate_by_time_of_day()`, `get_completion_rate_by_task_type()`, `get_avg_energy_by_completion()`, `get_avg_distraction_by_completion()`, `get_completion_summary()`, `get_user_summary()`
- Deeper analytical queries: `get_high_distraction_sessions()`, `get_top_performers()`, `get_session_duration_analysis()`, `get_focus_energy_by_task_time()`

---

### `analytics/productivity.py`
Core metrics computed with pandas and NumPy.

- `completion_rate_by_time_of_day(df)` — completion rate for morning / afternoon / night
- `completion_rate_by_task_type(df)` — completion rate for deep_work / creative / admin
- `correlations(df)` — Pearson correlations between energy, focus, distractions and completion. Also builds a full 5×5 correlation matrix using `numpy.corrcoef`
- `productivity_score(df)` — per-row weighted score (0–1) combining completion, energy, focus, and distraction penalty
- `full_report(df)` — runs all of the above and returns a single dict

---

### `scoring/engine.py`
Two per-row scores computed for every session.

- **Productivity Index (0–100)** — holistic session quality. Weights: completion 35%, energy 25%, focus 25%, distraction penalty 15%.
- **Cognitive Load Score (0–100)** — mental effort demanded. Based on task type base load (deep_work=70, creative=55, admin=35) plus duration and distraction modifiers.
- `_percentile_stats()` — uses `numpy.percentile` to compute P10/P25/P50/P75/P90 distribution of scores, giving a fuller picture than mean alone.
- `score_dataset(df)` — scores every row and returns summaries, band breakdowns (Excellent / Good / Moderate / Low / Poor), and breakdowns by task type and time of day.

---

### `patterns/detector.py`
Detects four behavioral patterns in the data. Each function returns a human-readable `insight` string plus supporting `details`.

- `best_time_of_day(df)` — which time slot has the highest completion rate and by how much
- `best_task_type_when_energy_high(df)` — which task type performs best when energy is high (≥4)
- `distraction_impact(df)` — how completion rate changes across distraction buckets (none / low / medium / high)
- `flow_state_analysis(df)` — detects "flow" sessions (focus ≥4 AND completed), reports how common they are and what conditions produce them most
- `all_patterns(df)` — runs all four and returns a combined dict

---

### `analytics/segmentation.py`
Groups users into performance segments and compares their behavior.

- `segment_users(df)` — classifies each user as High Performer (≥45% completion), Mid Performer (25–45%), or Low Performer (<25%). Requires at least 5 sessions to classify.
- `compare_segments(df)` — aggregates energy, focus, distractions, duration, and preferred task per segment
- `segment_insights(df)` — generates plain-language comparison strings (e.g. "High Performers average 3.66/5 energy vs 3.10/5 for Low Performers")
- `significance_tests(df)` — runs independent-samples t-tests (`scipy.stats.ttest_ind`) on energy, focus, and distraction count between High and Low Performers. Returns p-values and whether the difference is statistically significant (p < 0.05).
- `full_segmentation_report(df)` — combines all four outputs

---

### `analytics/recommendations.py`
Generates recommendations purely from computed data — no fixed rules or hardcoded strings.

- `best_time_of_day(df)` — finds the time slot with the highest completion rate and quantifies the gap
- `best_task_by_energy(df)` — for each energy band (low / medium / high), finds the task type with the best completion rate
- `distraction_threshold(df)` — finds the exact distraction count where completion rate drops below 25% using a rolling average to smooth noise
- `generate_recommendations(df)` — combines the three computations above into 5–6 actionable recommendation strings, every number pulled from the data
- `full_recommendations_report(df)` — returns all computations plus the recommendation list

---

### `analytics/visualizations.py`
Renders matplotlib charts to base64-encoded PNG strings. No files are written to disk — charts are returned in the API response and displayed with `<img>` tags in the frontend.

- `bar_completion_by_time_of_day(df)` — bar chart of completion rate for morning / afternoon / night
- `heatmap_energy_vs_completion(df)` — heatmap of completion rate across all energy level (1–5) × task type combinations
- `generate_all_charts(df)` — renders both and returns a dict of base64 strings

---

### `psychology/mapper.py`
Maps the data patterns to three established psychology concepts. Each function returns a structured dict with the concept name, the research theory behind it, what the data shows, why it happens, and one actionable recommendation.

- `cognitive_load_profile(df)` — maps task complexity, session length, and distractions to **Cognitive Load Theory** (Sweller, 1988). Working memory has a fixed capacity; exceeding it causes performance to collapse.
- `attention_span_profile(df)` — maps focus degradation and distraction impact to **Directed Attention Fatigue** (Kaplan, 1995). Voluntary focus is a finite resource that depletes through use.
- `energy_cycle_profile(df)` — maps energy levels and time-of-day patterns to **Circadian Rhythms and Ultradian Performance Cycles** (Kleitman, 1963). The body has predictable peaks and troughs in alertness throughout the day.
- `behavioral_explanation(df)` — synthesises all three profiles into a plain-language summary, list of observed behaviors, root causes, and a prioritised action plan.
- `full_psychology_report(df)` — runs all four and returns a combined dict.

---

## Frontend components

The Angular app has one component per dashboard section. All components receive their data as `@Input()` from the root `App` component, which fetches everything in a single cached HTTP call.

| Component | What it shows |
|---|---|
| `MetricsComponent` | SVG ring charts for Productivity Score and top completion rate, plus bar breakdowns by time of day and task type |
| `InsightsComponent` | Carousel of pattern insights from `patterns/detector.py` and psychology observations |
| `RecommendationsComponent` | Carousel of psychology-based action items |
| `SegmentationComponent` | High / Mid / Low Performer cards with behavioral stats and the significance test insights |
| `DataRecommendationsComponent` | Key stats strip (best time, distraction threshold) and carousel of data-computed recommendations |
| `PsychologyComponent` | Behavioral summary from the psychology mapper |
| `AIComponent` | Button that calls `/grok` and displays the Groq-generated coaching insight |
| `ChartsComponent` | The two matplotlib charts rendered as images |

---

## Data schema

Each row in `productivity.csv` (and the SQLite table) represents one work session:

| Column | Type | Description |
|---|---|---|
| `user_id` | text | User identifier (e.g. U039) |
| `task_type` | text | `deep_work`, `creative`, or `admin` |
| `session_start` | text | Session start timestamp |
| `session_end` | text | Session end timestamp |
| `duration` | integer | Session length in minutes |
| `time_of_day` | text | `morning`, `afternoon`, or `night` |
| `energy_level` | integer | Self-reported energy 1–5 |
| `focus_level` | integer | Self-reported focus 1–5 |
| `distraction_count` | integer | Number of distractions during session |
| `distraction_duration` | integer | Total minutes lost to distractions |
| `completed` | integer | 1 if task was completed, 0 if not |

---

## Key findings from the dataset

- Morning sessions complete at **47.7%** — nearly 3× the night rate of 16.9%
- Deep work with high energy (≥4) completes at **68.2%**
- Sessions with fewer than 7 distractions complete at **42.1%** — above 7 it drops to **12.0%**
- The energy difference between High and Low Performers is statistically significant (**p = 0.0003**)
- Focus is a stronger predictor of completion (r = 0.40) than energy (r = 0.33)
