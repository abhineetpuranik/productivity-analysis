import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { DataService } from './services/data';
import { AnalysisResponse } from './models';
import { MetricsComponent } from './components/metrics/metrics';
import { InsightsComponent } from './components/insights/insights';
import { RecommendationsComponent } from './components/recommendations/recommendations';
import { PsychologyComponent } from './components/psychology/psychology';
import { AIComponent } from './components/ai/ai';
import { ChartsComponent } from './components/charts/charts';
import { SegmentationComponent } from './components/segmentation/segmentation';
import { DataRecommendationsComponent } from './components/data-recommendations/data-recommendations';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    CommonModule,
    MetricsComponent,
    InsightsComponent,
    RecommendationsComponent,
    PsychologyComponent,
    AIComponent,
    ChartsComponent,
    SegmentationComponent,
    DataRecommendationsComponent,
  ],
  templateUrl: './app.html',
  styleUrl: './app.scss',
})
export class App implements OnInit {
  data: AnalysisResponse | null = null;
  loading = true;
  error: string | null = null;

  constructor(private dataService: DataService) {}

  ngOnInit(): void {
    this.dataService.getAnalysis().subscribe({
      next: res => {
        this.data = res;
        this.loading = false;
      },
      error: (err: Error) => {
        this.error = err.message;
        this.loading = false;
      },
    });
  }
}
