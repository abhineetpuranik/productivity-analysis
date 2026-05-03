import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { DataRecommendations } from '../../models';

@Component({
  selector: 'app-data-recommendations',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './data-recommendations.html',
  styleUrl: './data-recommendations.scss',
})
export class DataRecommendationsComponent {
  @Input() dataRecs: DataRecommendations | null = null;
  @Input() loading = false;
  @Input() error: string | null = null;

  current = 0;

  get recs(): string[] {
    return this.dataRecs?.recommendations ?? [];
  }

  get total(): number { return this.recs.length; }

  prev(): void {
    this.current = this.current > 0 ? this.current - 1 : this.total - 1;
  }

  next(): void {
    this.current = this.current < this.total - 1 ? this.current + 1 : 0;
  }

  goTo(i: number): void { this.current = i; }

  pct(value: number): string {
    return (value * 100).toFixed(1) + '%';
  }
}
