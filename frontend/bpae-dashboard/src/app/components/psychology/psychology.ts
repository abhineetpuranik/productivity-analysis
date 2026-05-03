import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Behavior } from '../../models';

@Component({
  selector: 'app-psychology',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './psychology.html',
  styleUrl: './psychology.scss',
})
export class PsychologyComponent {
  @Input() behavior: Behavior | null = null;
  @Input() loading = false;
  @Input() error: string | null = null;
}
