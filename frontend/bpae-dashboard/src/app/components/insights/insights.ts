import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-insights',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './insights.html',
  styleUrl: './insights.scss',
})
export class InsightsComponent {
  @Input() set insights(value: string[]) {
    this._insights = value;
    this.current = 0;
  }
  get insights(): string[] { return this._insights; }

  private _insights: string[] = [];
  current = 0;
  loading = false;
  @Input() error: string | null = null;

  get total(): number { return this._insights.length; }

  prev(): void {
    this.current = this.current > 0 ? this.current - 1 : this.total - 1;
  }

  next(): void {
    this.current = this.current < this.total - 1 ? this.current + 1 : 0;
  }

  goTo(i: number): void { this.current = i; }
}
