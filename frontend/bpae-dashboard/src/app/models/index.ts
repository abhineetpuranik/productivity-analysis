// ── Metrics ──────────────────────────────────────────────────────────────────

export interface CompletionRate {
  by_time_of_day: Record<string, number>;
  by_task_type:   Record<string, number>;
}

export interface CorrelationEntry {
  pearson:         number;
  point_biserial:  number;
  interpretation:  string;
}

export interface ProductivityScoreSummary {
  mean:           number;
  median:         number;
  std:            number;
  min:            number;
  max:            number;
  by_task_type:   Record<string, number>;
  by_time_of_day: Record<string, number>;
}

export interface ProductivityIndex {
  summary:        { mean: number; median: number; std: number; min: number; max: number };
  by_task_type:   Record<string, number>;
  by_time_of_day: Record<string, number>;
}

export interface CognitiveLoadScore {
  summary: { mean: number; median: number; std: number; min: number; max: number };
}

export interface Metrics {
  completion_rate:       CompletionRate;
  correlations:          Record<string, CorrelationEntry>;
  productivity_score:    ProductivityScoreSummary;
  productivity_index:    ProductivityIndex;
  cognitive_load_score:  CognitiveLoadScore;
  score_weights:         Record<string, number>;
}

// ── Behavior ─────────────────────────────────────────────────────────────────

export interface Behavior {
  summary:       string;
  key_behaviors: string[];
  root_causes:   string[];
  action_plan:   string[];
}

// ── AI Insight ───────────────────────────────────────────────────────────────

export interface AIInsight {
  text: string;
}

// ── Charts ────────────────────────────────────────────────────────────────────

export interface Charts {
  bar_completion_by_time_of_day: string;  // base64-encoded PNG
  heatmap_energy_vs_completion:  string;  // base64-encoded PNG
}

// ── Segmentation ──────────────────────────────────────────────────────────────

export interface UserSegment {
  user_id:          string;
  total_sessions:   number;
  completion_rate:  number;
  avg_energy:       number;
  avg_focus:        number;
  avg_distractions: number;
  avg_duration:     number;
  preferred_task:   string;
  segment:          string;
}

export interface SegmentStats {
  user_count:          number;
  session_count:       number;
  avg_completion_rate: number;
  avg_energy:          number;
  avg_focus:           number;
  avg_distractions:    number;
  avg_duration:        number;
  preferred_task:      string;
}

export interface Segmentation {
  user_segments:       UserSegment[];
  segment_comparison:  Record<string, SegmentStats>;
  insights:            string[];
}

// ── Data Recommendations ──────────────────────────────────────────────────────

export interface DataRecommendations {
  best_time_of_day: {
    best_slot:        string;
    completion_rates: Record<string, number>;
    lift_over_worst:  number;
    session_counts:   Record<string, number>;
  };
  best_task_by_energy: Record<string, {
    best_task:       string;
    completion_rate: number;
    session_count:   number;
  }>;
  distraction_threshold: {
    threshold:             number | null;
    performance_floor:     number;
    rate_below_threshold:  number | null;
    rate_above_threshold:  number | null;
  };
  recommendations: string[];
}

// ── Full API response ─────────────────────────────────────────────────────────

export interface AnalysisResponse {
  metrics:                 Metrics;
  insights:                string[];
  recommendations:         string[];
  data_recommendations:    DataRecommendations;
  behavioral_explanation:  Behavior;
  segmentation:            Segmentation;
  charts:                  Charts;
  ai_prompt:               string;
}
