"""
Particle Swarm Optimization (from scratch) to tune the CogWear focus XGBoost model.

Objective : maximize LOSO AUC on CogWear (per-subject z-normalized).
Guard     : after tuning, re-check the best model zero-shot on MAUS (untouched) so we
            can see whether the gains generalize or just overfit the validation.

Run: python3 ml/focus/src/pso_optimize.py [n_particles] [n_iters]
"""
import os, sys, glob, json, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'stress'))
from src.features import compute_time_domain, compute_nonlinear  # noqa
import xgboost as xgb
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FEAT = ['meanRR','sdnn','rmssd','pnn50','pnn20','hrMean','hrStd','hrRange','cvRR',
        'sd1','sd2','sd1sd2Ratio','sampleEntropy','dfaAlpha1']
NB = {1: 0, 2: 2, 3: 3, 4: 2, 5: 3, 6: 0}
np.random.seed(42)


def per_subject_z(df):
    o = df.copy()
    for f in FEAT:
        g = o.groupby('subject_id')[f]
        o[f] = (o[f] - g.transform('mean')) / g.transform('std').replace(0, 1.0).fillna(1.0)
    return o.fillna(0.0)


def hrv(rr):
    rr = np.asarray(rr, float); rr = rr[(rr >= 300) & (rr <= 2000)]
    if len(rr) > 5:
        m = np.median(rr); rr = rr[np.abs(rr - m) / m < 0.30]
    if len(rr) < 20:
        return None
    d = {}; d.update(compute_time_domain(rr)); d.update(compute_nonlinear(rr))
    return {k: float(d.get(k, 0.0)) for k in FEAT}


def load_maus():
    root = os.path.join(BASE, 'focus', 'data', 'maus', 'MAUS', 'Data', 'IBI_sequence')
    rows = []
    for sub in sorted(os.listdir(root)):
        sd = os.path.join(root, sub)
        if not os.path.isdir(sd):
            continue
        for f in glob.glob(os.path.join(sd, '*.csv')):
            n = os.path.basename(f)[:-4]
            if 'peak' in n:
                continue
            cond = 'rest' if n.startswith('rest') else (f'{NB[int(n.split("_")[1])]}back' if n.startswith('trial_') else None)
            if cond not in ('rest', '2back', '3back'):
                continue
            try:
                d = pd.read_csv(f)
            except Exception:
                continue
            if 'RRI_inf' not in d.columns:
                continue
            fe = hrv(d['RRI_inf'].values)
            if fe:
                rows.append({'subject_id': f'M_{sub}', 'label': int(cond != 'rest'), **fe})
    return per_subject_z(pd.DataFrame(rows))


# ---- data (normalize once) ----
cog = per_subject_z(pd.read_csv(os.path.join(BASE, 'focus', 'data', 'cogwear_features_samsung.csv')))
Xc, yc, gc = cog[FEAT].values, cog['label'].values.astype(int), cog['subject_id'].values
maus = load_maus()
Xm, ym = maus[FEAT].values, maus['label'].values.astype(int)
_logo = LeaveOneGroupOut()
_folds = list(_logo.split(Xc, yc, gc))


def loso_auc(params):
    yt, yp = [], []
    for tr, te in _folds:
        pos = max(1, (yc[tr] == 1).sum()); neg = max(1, (yc[tr] == 0).sum())
        m = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                              scale_pos_weight=neg / pos, random_state=42, n_jobs=-1, **params)
        m.fit(Xc[tr], yc[tr])
        yp.extend(m.predict_proba(Xc[te])[:, 1]); yt.extend(yc[te])
    return roc_auc_score(yt, yp)


def maus_auc(params):
    pos = max(1, (yc == 1).sum()); neg = max(1, (yc == 0).sum())
    m = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                          scale_pos_weight=neg / pos, random_state=42, n_jobs=-1, **params)
    m.fit(Xc, yc)
    return roc_auc_score(ym, m.predict_proba(Xm)[:, 1])


# ---- search space: (low, high, is_int) ----
SPACE = {
    'max_depth':        (2, 5, True),
    'learning_rate':    (0.01, 0.20, False),
    'n_estimators':     (100, 450, True),
    'subsample':        (0.6, 1.0, False),
    'colsample_bytree': (0.6, 1.0, False),
    'reg_lambda':       (0.0, 5.0, False),
    'reg_alpha':        (0.0, 2.0, False),
    'gamma':            (0.0, 1.0, False),
    'min_child_weight': (1, 8, True),
}
KEYS = list(SPACE.keys())
LO = np.array([SPACE[k][0] for k in KEYS], float)
HI = np.array([SPACE[k][1] for k in KEYS], float)


def decode(pos):
    vals = LO + pos * (HI - LO)
    out = {}
    for i, k in enumerate(KEYS):
        out[k] = int(round(vals[i])) if SPACE[k][2] else float(vals[i])
    return out


def main():
    n_part = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    n_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    D = len(KEYS)

    # baseline (current shipped params)
    base = dict(max_depth=2, learning_rate=0.05, n_estimators=400, subsample=0.8,
                colsample_bytree=0.8, reg_lambda=2.0, reg_alpha=0.5, gamma=0.2, min_child_weight=1)
    base_auc = loso_auc(base)
    print(f"baseline (shipped) CogWear LOSO AUC = {base_auc:.4f}", flush=True)

    pos = np.random.rand(n_part, D)
    vel = (np.random.rand(n_part, D) - 0.5) * 0.2
    pbest = pos.copy(); pbest_f = np.full(n_part, -1.0)
    gbest = None; gbest_f = -1.0
    w, c1, c2 = 0.7, 1.5, 1.5

    for it in range(n_iter):
        for i in range(n_part):
            f = loso_auc(decode(pos[i]))
            if f > pbest_f[i]:
                pbest_f[i] = f; pbest[i] = pos[i].copy()
            if f > gbest_f:
                gbest_f = f; gbest = pos[i].copy()
        # update
        r1, r2 = np.random.rand(n_part, D), np.random.rand(n_part, D)
        vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        vel = np.clip(vel, -0.3, 0.3)
        pos = np.clip(pos + vel, 0.0, 1.0)
        print(f"  iter {it+1:2d}/{n_iter}  best LOSO AUC so far = {gbest_f:.4f}", flush=True)

    best = decode(gbest)
    best_loso = gbest_f
    best_maus = maus_auc(best)
    base_maus = maus_auc(base)

    print("\n================ PSO RESULT ================")
    print(f"baseline : CogWear LOSO AUC {base_auc:.4f} | MAUS external AUC {base_maus:.4f}")
    print(f"optimized: CogWear LOSO AUC {best_loso:.4f} | MAUS external AUC {best_maus:.4f}")
    print("best params:", json.dumps(best))
    out = os.path.join(BASE, 'focus', 'models', 'pso_best_params.json')
    json.dump({'best_params': best, 'cogwear_loso_auc': best_loso, 'maus_external_auc': best_maus,
               'baseline_loso_auc': base_auc, 'baseline_maus_auc': base_maus}, open(out, 'w'), indent=2)
    print("saved ->", out)


if __name__ == '__main__':
    main()
