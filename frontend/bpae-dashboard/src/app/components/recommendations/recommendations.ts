import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

const ICONS = ['🎯', '⏱️', '🔕', '📅', '🌿'];

@Component({
  selector: 'app-recommendations',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './recommendations.html',
  styleUrl: './recommendations.scss',
})
export class RecommendationsComponent {
  @Input() set recommendations(value: string[]) {
    this._recs = value;
    this.current = 0;
  }
  get recommendations(): string[] { return this._recs; }

  private _recs: string[] = [];
  current = 0;
  @Input() loading = false;
  @Input() error: string | null = null;

  get total(): number { return this._recs.length; }

  icon(i: number): string { return ICONS[i % ICONS.length]; }

  isTop(i: number): boolean { return i === 0; }

  prev(): void {
    this.current = this.current > 0 ? this.current - 1 : this.total - 1;
  }

  next(): void {
    this.current = this.current < this.total - 1 ? this.current + 1 : 0;
  }

  goTo(i: number): void { this.current = i; }
}
