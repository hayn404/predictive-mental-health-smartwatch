import type { BiometricFeatureVector } from './types';

export interface PerUserNorm {
  mean: Record<string, number>;
  std: Record<string, number>;
  sampleCount: number;
}

/**
 * Per-subject (per-user) normalization — the generalization lever the model was trained
 * with. Instead of the model's GLOBAL training mean/std, z-score each feature against the
 * USER's own recent-window distribution (matching the per-subject z-norm used to reach the
 * reported cross-dataset AUC).
 *
 * Returns null until there are at least `minSamples` windows (cold-start) so the caller
 * falls back to global normalization. Features without enough usable samples are omitted,
 * and the caller falls back to global for just those features.
 */
export function computePerUserNorm(
  windows: BiometricFeatureVector[] | null | undefined,
  featureNames: string[],
  minSamples = 30,
): PerUserNorm | null {
  if (!windows || windows.length < minSamples) return null;

  const mean: Record<string, number> = {};
  const std: Record<string, number> = {};

  for (const name of featureNames) {
    const vals: number[] = [];
    for (const w of windows) {
      const v = (w as any)[name];
      if (typeof v === 'number' && Number.isFinite(v)) vals.push(v);
    }
    if (vals.length < minSamples) continue; // fall back to global for this feature

    const m = vals.reduce((s, x) => s + x, 0) / vals.length;
    const variance = vals.reduce((s, x) => s + (x - m) * (x - m), 0) / vals.length;
    mean[name] = m;
    std[name] = Math.sqrt(variance);
  }

  return { mean, std, sampleCount: windows.length };
}
