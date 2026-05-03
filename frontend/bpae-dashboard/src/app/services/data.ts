import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError, map, shareReplay, timeout } from 'rxjs/operators';
import { AIInsight, AnalysisResponse } from '../models';

const API_URL  = 'http://localhost:5000/analyze';
const GROQ_URL = 'http://localhost:5000/groq';

// Groq can take a few seconds — allow up to 30s before showing an error
const GROQ_TIMEOUT_MS = 30_000;

@Injectable({ providedIn: 'root' })
export class DataService {
  private analysis$: Observable<AnalysisResponse> | null = null;

  constructor(private http: HttpClient) {}

  /**
   * GET /analyze – fetches the full analysis payload.
   * Result is cached via shareReplay(1) so all components share one request.
   * Call refresh() to force a new fetch.
   */
  getAnalysis(): Observable<AnalysisResponse> {
    if (!this.analysis$) {
      this.analysis$ = this.http
        .get<AnalysisResponse>(API_URL)
        .pipe(
          shareReplay(1),
          catchError(this.handleError),
        );
    }
    return this.analysis$;
  }

  /** Clears the cache so the next getAnalysis() hits the server fresh. */
  refresh(): void {
    this.analysis$ = null;
  }

  /**
   * GET /grok – sends the analysis prompt to Groq and returns the response.
   * Uncached; each call hits the model fresh.
   */
  getGroqInsight(): Observable<AIInsight> {
    return this.http
      .get<{ insight: string }>(GROQ_URL)
      .pipe(
        timeout(GROQ_TIMEOUT_MS),
        map(res => ({ text: res.insight })),
        catchError(this.handleError),
      );
  }

  private handleError(error: HttpErrorResponse): Observable<never> {
    let message: string;

    if (error.status === 0) {
      // Network error – server not reachable
      message = 'Unable to reach the server. Make sure the backend is running on localhost:5000.';
    } else {
      // Try to extract the "error" field from the JSON body first,
      // then fall back to the HTTP status message.
      const body = error.error;
      const serverMsg: string =
        (typeof body === 'object' && body !== null && typeof body['error'] === 'string')
          ? body['error']
          : error.message;

      // Trim full tracebacks – show only the last meaningful line
      const lines = serverMsg.split('\n').map(l => l.trim()).filter(Boolean);
      const shortMsg = lines[lines.length - 1] ?? serverMsg;

      message = `Error ${error.status}: ${shortMsg}`;
    }

    return throwError(() => new Error(message));
  }
}
