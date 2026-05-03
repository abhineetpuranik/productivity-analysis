import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Metrics } from '../../models';

const CIRCUMFERENCE = 2 * Math.PI * 50; // r = 50 → ≈ 314.16

@Component({
  selector: 'app-metrics',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './metrics.html',
  styleUrl: './metrics.scss',
})
export class MetricsComponent {
  @Input() metrics: Metrics | null = null;
  @Input() loading = false;
  @Input() error: string | null = null;

  /** Best time-of-day key by completion rate. */
  bestTimeOfDay(m: Metrics): string {
    const tod = m.completion_rate.by_time_of_day;
    return Object.keys(tod).reduce((a, b) => tod[a] > tod[b] ? a : b);
  }

  /** Best task type key by completion rate. */
  bestTaskType(m: Metrics): string {
    const tt = m.completion_rate.by_task_type;
    return Object.keys(tt).reduce((a, b) => tt[a] > tt[b] ? a : b);
  }

  /** Format a 0–1 rate as a percentage string. */
  pct(value: number): string {
    return (value * 100).toFixed(1) + '%';
  }

  /** Format a 0–1 score as 0–100. */
  score100(value: number): string {
    return (value * 100).toFixed(0);
  }

  /** SVG ring stroke-dashoffset for a 0–1 value. */
  ringOffset(value: number): number {
    const pct = Math.min(Math.max(value, 0), 1);
    return CIRCUMFERENCE * (1 - pct);
  }

  objectKeys(obj: Record<string, unknown>): string[] {
    return Object.keys(obj);
  }
}
