/**
 * Seren — Biological / Physiological Age Model (On-Device Inference)
 * =================================================================
 * Pure-TS XGBoost tree traversal (same engine as the stress/focus models), but a
 * REGRESSOR: predicts physiological age (years) from HRV. No sigmoid.
 *
 * Unlike focus/stress (per-user normalization), age is a BETWEEN-subject trait, so we
 * use the model's GLOBAL normalization. Age is also stable, so we aggregate the user's
 * recent windows (median) before predicting — matching the training representation.
 *
 * Output: predicted age + age-gap (predicted − chronological). A positive gap means the
 * body is "older than its years" — accelerated by chronic stress / poor recovery.
 *
 * Validation (PhysioNet Autonomic Aging, 1095 subjects): LOSO MAE 7.6 yrs, r 0.68;
 * cross-dataset young-vs-old AUC 0.94.
 */

import { BiometricFeatureVector } from './types';

interface XGBoostNode {
  nodeid: number;
  split?: string;
  split_condition?: number;
  yes?: number;
  no?: number;
  missing?: number;
  leaf?: number;
  children?: XGBoostNode[];
}

interface BioAgeModel {
  version: string;
  task?: string;
  features: string[];
  baseScore: number;
  applySigmoid?: boolean;
  normalization: { mean: Record<string, number>; std: Record<string, number> };
  trees: XGBoostNode[];
}

export interface BioAgePrediction {
  timestamp: number;
  predictedAge: number;            // physiological age (years)
  chronologicalAge: number | null;
  ageGap: number | null;           // predicted − chronological
  confidence: number;              // 0–1, grows with amount of personal data
  ready: boolean;                  // enough windows for a stable estimate
}

const MIN_WINDOWS = 10;            // need a bit of history for a stable median
let cached: BioAgeModel | null = null;

export function loadBioAgeModel(json: BioAgeModel): void {
  cached = json;
}

export function isBioAgeModelLoaded(): boolean {
  return cached !== null && Array.isArray(cached.features);
}

// ── Tree traversal ────────────────────────────────────────────
function traverse(node: XGBoostNode, f: Record<string, number>): number {
  if (node.leaf !== undefined) return node.leaf;
  const v = f[node.split!];
  if (v === undefined || v === null || isNaN(v)) {
    const id = node.missing ?? node.no!;
    const c = node.children?.find(x => x.nodeid === id);
    return c ? traverse(c, f) : 0;
  }
  const id = v < node.split_condition! ? node.yes! : node.no!;
  const c = node.children?.find(x => x.nodeid === id);
  return c ? traverse(c, f) : 0;
}

function median(arr: number[]): number {
  if (!arr.length) return 0;
  const s = [...arr].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/**
 * Predict physiological age from the user's recent feature windows.
 * @param recentWindows  rolling history of live HRV windows
 * @param chronologicalAge  the user's real age (for the gap); null if unknown
 */
export function predictBioAge(
  recentWindows: BiometricFeatureVector[] | null | undefined,
  chronologicalAge: number | null = null,
): BioAgePrediction {
  const now = Date.now();
  const notReady = (): BioAgePrediction => ({
    timestamp: now, predictedAge: chronologicalAge ?? 0, chronologicalAge,
    ageGap: null, confidence: 0, ready: false,
  });

  if (!isBioAgeModelLoaded() || !recentWindows || recentWindows.length < MIN_WINDOWS) {
    return notReady();
  }

  // Use windows with a usable cardiac signal only.
  const usable = recentWindows.filter(w => w.rmssd > 0 && w.meanRR > 0);
  if (usable.length < MIN_WINDOWS) return notReady();

  // Subject-aggregate: median per feature (matches training).
  const norm: Record<string, number> = {};
  for (const name of cached!.features) {
    const vals = usable.map(w => (w as any)[name]).filter(v => typeof v === 'number' && isFinite(v));
    const med = median(vals);
    const mean = cached!.normalization.mean[name] ?? 0;
    const std = cached!.normalization.std[name] ?? 1;
    norm[name] = std > 0 ? (med - mean) / std : 0;
  }

  let pred = cached!.baseScore;
  for (const t of cached!.trees) pred += traverse(t, norm);
  const predictedAge = Math.round(Math.max(15, Math.min(95, pred)));

  const ageGap = chronologicalAge != null ? Math.round(predictedAge - chronologicalAge) : null;
  // Confidence ramps with how much personal data backs the median (cap ~1 day of windows).
  const confidence = Math.min(1, usable.length / 60);

  return { timestamp: now, predictedAge, chronologicalAge, ageGap, confidence, ready: true };
}

/** Short human description of the age gap for the UI. */
export function ageGapMessage(p: BioAgePrediction): string {
  if (!p.ready) return 'Gathering enough signal to estimate your physiological age…';
  if (p.ageGap == null) return `Your physiological age estimate is ${p.predictedAge}.`;
  if (p.ageGap <= -3) return `Your body looks ${Math.abs(p.ageGap)} yrs younger than your age — strong recovery.`;
  if (p.ageGap >= 3) return `Your body looks ${p.ageGap} yrs older than your age — stress and recovery may be a factor.`;
  return 'Your physiological age is right around your actual age.';
}
