import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Segmentation, SegmentStats } from '../../models';

@Component({
  selector: 'app-segmentation',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './segmentation.html',
  styleUrl: './segmentation.scss',
})
export class SegmentationComponent {
  @Input() segmentation: Segmentation | null = null;
  @Input() loading = false;
  @Input() error: string | null = null;

  get segments(): string[] {
    if (!this.segmentation) return [];
    return Object.keys(this.segmentation.segment_comparison);
  }

  stats(segment: string): SegmentStats | null {
    return this.segmentation?.segment_comparison[segment] ?? null;
  }

  pct(value: number): string {
    return (value * 100).toFixed(1) + '%';
  }

  segmentColor(segment: string): string {
    if (segment.includes('High')) return 'high';
    if (segment.includes('Low'))  return 'low';
    return 'mid';
  }
}
