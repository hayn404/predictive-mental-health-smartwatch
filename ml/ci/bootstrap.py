"""
Bootstrap confidence intervals for model metrics.

For a metric computed on held-out predictions, resample (with replacement)
`n_boot` times and take the 2.5th/97.5th percentiles -> 95% CI. When `groups`
is given (e.g. subject ids), whole groups are resampled (subject-level cluster
bootstrap) so the interval reflects between-subject variability rather than the
artificially-tight within-subject one. Deterministic (seeded).
"""
import numpy as np


def bootstrap_ci(metric_fn, y_true, y_score, groups=None, n_boot=1000, seed=42, alpha=0.05):
    """Return (lo, hi) 95% CI for metric_fn(y_true[idx], y_score[idx])."""
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    if groups is not None:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        idx_by_g = {g: np.where(groups == g)[0] for g in uniq}

    vals = []
    for _ in range(n_boot):
        if groups is not None:
            chosen = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([idx_by_g[g] for g in chosen])
        else:
            idx = rng.randint(0, n, n)
        try:
            v = float(metric_fn(y_true[idx], y_score[idx]))
        except Exception:
            continue
        if v == v:  # drop NaN (e.g. a resample with one class)
            vals.append(v)
    if not vals:
        return [float("nan"), float("nan")]
    lo = float(np.percentile(vals, 100 * alpha / 2))
    hi = float(np.percentile(vals, 100 * (1 - alpha / 2)))
    return [round(lo, 4), round(hi, 4)]
