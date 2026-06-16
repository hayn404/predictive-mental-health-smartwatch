"""
Finalize the biological-age model:
  1. Compare window-level vs subject-aggregate representation (age is a stable trait →
     averaging a subject's windows usually gives a cleaner estimate).
  2. Light hyperparameter pick on AA LOSO.
  3. Train the FINAL model on all AA, export it (+ global-norm stats + metadata).
  4. Cross-dataset test on Fantasia: raw + honest split-half recalibration.
  5. Define the age-gap metric.

Run: python3 ml/bioage/src/finalize_bioage.py
"""
import os, json, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
import xgboost as xgb
from sklearn.model_selection import LeaveOneGroupOut, KFold
from sklearn.metrics import mean_absolute_error, r2_score

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FEAT = ['meanRR','sdnn','rmssd','pnn50','pnn20','hrMean','cvRR','sd1','sd2','sd1sd2Ratio','sampleEntropy','dfaAlpha1']


def load(name):
    df = pd.read_csv(os.path.join(BASE, 'data', f'bioage_features_{name}.csv')).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def aggregate(df):
    """One row per subject = median of their windows (robust)."""
    g = df.groupby('subject_id')
    agg = g[FEAT].median()
    agg['age'] = g['age'].first()
    return agg.reset_index()


def gnorm(X, m=None, s=None):
    if m is None:
        m, s = X.mean(0), X.std(0) + 1e-9
    return (X - m) / s, m, s


def metrics(yt, yp):
    return mean_absolute_error(yt, yp), np.corrcoef(yt, yp)[0, 1], r2_score(yt, yp)


def loo_agg(df, params):
    """Subject-level 10-fold CV on the aggregate table (1 row/subject → no leakage)."""
    X = df[FEAT].values; y = df['age'].values.astype(float)
    yp = np.zeros_like(y)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for tr, te in kf.split(X):
        Xtr, m, s = gnorm(X[tr]); Xte, _, _ = gnorm(X[te], m, s)
        reg = xgb.XGBRegressor(**params); reg.fit(Xtr, y[tr])
        yp[te] = reg.predict(Xte)
    return metrics(y, yp)


def main():
    aa = load('aa'); fan = load('fantasia')
    aa_ag = aggregate(aa); fan_ag = aggregate(fan)
    print(f"AA: {aa.subject_id.nunique()} subjects | Fantasia: {fan.subject_id.nunique()} subjects\n")

    # ---- 1+2: representation + light tuning on AA (subject-aggregate, LOO) ----
    configs = {
        'depth3_n400_lr03': dict(n_estimators=400, learning_rate=0.03, max_depth=3),
        'depth4_n600_lr03': dict(n_estimators=600, learning_rate=0.03, max_depth=4),
        'depth2_n800_lr02': dict(n_estimators=800, learning_rate=0.02, max_depth=2),
    }
    common = dict(objective='reg:squarederror', subsample=0.8, colsample_bytree=0.8,
                  reg_lambda=2.0, reg_alpha=0.5, random_state=42, n_jobs=-1)
    best = None
    print("AA subject-aggregate LOO (each subject = median of windows):")
    for name, p in configs.items():
        mae, r, r2 = loo_agg(aa_ag, {**common, **p})
        print(f"  {name:20s} MAE={mae:.2f} yrs  r={r:.3f}  R2={r2:.3f}")
        if best is None or r > best[1]:
            best = (mae, r, r2, name, {**common, **p})
    print(f"  -> best: {best[3]} (MAE {best[0]:.2f}, r {best[1]:.3f})\n")
    params = best[4]

    # ---- 3: train FINAL model on all AA + export ----
    X = aa_ag[FEAT].values; y = aa_ag['age'].values.astype(float)
    Xn, m, s = gnorm(X)
    final = xgb.XGBRegressor(**params); final.fit(Xn, y)
    mdir = os.path.join(BASE, 'models'); os.makedirs(mdir, exist_ok=True)
    final.get_booster().save_model(os.path.join(mdir, 'bioage_model.json'))
    meta = {'features': FEAT, 'norm_mean': m.tolist(), 'norm_std': s.tolist(),
            'representation': 'subject-aggregate (median of 120s HRV windows)',
            'train_data': 'Autonomic Aging (PhysioNet)', 'n_subjects': int(aa_ag.subject_id.nunique()),
            'loso_mae': best[0], 'loso_r': best[1], 'params': params,
            'note': 'age = group midpoint; global normalization; output age-gap = predicted - chronological'}
    json.dump(meta, open(os.path.join(mdir, 'bioage_model_meta.json'), 'w'), indent=2)
    print(f"exported FINAL model -> ml/bioage/models/bioage_model.json (+ meta)\n")

    # ---- 4: cross-dataset Fantasia (raw + honest split-half recalibration) ----
    Xf, _, _ = gnorm(fan_ag[FEAT].values, m, s); yf = fan_ag['age'].values.astype(float)
    pf = final.predict(Xf)
    mae_raw, r_raw, _ = metrics(yf, pf)
    # honest recalibration: fit affine (a*pred+b) on half the Fantasia subjects, test on other half
    idx = np.arange(len(yf)); rng = np.random.RandomState(0); rng.shuffle(idx)
    h = len(idx) // 2; cal, tst = idx[:h], idx[h:]
    a, b = np.polyfit(pf[cal], yf[cal], 1)
    mae_cal = mean_absolute_error(yf[tst], a * pf[tst] + b)
    print("Cross-dataset AA -> Fantasia (different device + population):")
    print(f"  raw:               MAE={mae_raw:.2f} yrs  r={r_raw:.3f}")
    print(f"  after per-device recalibration (split-half, honest): MAE={mae_cal:.2f} yrs")
    print("  (r is the generalization evidence; recalibration fixes the absolute offset)\n")

    # ---- 5: age-gap ----
    print("AGE-GAP metric = predicted_age - chronological_age")
    print("  >0 = aging faster (stress/poor recovery) ; <0 = younger-than-age")


if __name__ == '__main__':
    main()
