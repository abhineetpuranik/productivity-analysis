import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Charts } from '../../models';

@Component({
  selector: 'app-charts',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './charts.html',
  styleUrl: './charts.scss',
})
export class ChartsComponent {
  @Input() charts: Charts | null = null;
  @Input() loading = false;
  @Input() error: string | null = null;

  /** Prefix a raw base64 string with the data URI scheme. */
  toSrc(b64: string): string {
    return `data:image/png;base64,${b64}`;
  }
}
