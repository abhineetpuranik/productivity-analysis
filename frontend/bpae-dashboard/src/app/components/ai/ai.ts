import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { DataService } from '../../services/data';
import { AIInsight } from '../../models';

type State = 'idle' | 'loading' | 'success' | 'error';

@Component({
  selector: 'app-ai',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './ai.html',
  styleUrl: './ai.scss',
})
export class AIComponent {
  state: State = 'idle';
  generated: AIInsight | null = null;
  error: string | null = null;

  constructor(private data: DataService) {}

  generate(): void {
    if (this.state === 'loading') return;
    this.state = 'loading';
    this.generated = null;
    this.error = null;

    this.data.getGroqInsight().subscribe({
      next: insight => {
        this.generated = insight;
        this.state = 'success';
      },
      error: (err: Error) => {
        this.error = err.message;
        this.state = 'error';
      },
    });
  }
}
