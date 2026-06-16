"""
External validation of the DEPLOYED focus model.

Ship: CogWear-only model (trained on rest vs Stroop cognitive task).
Test: MAUS — a completely independent wearable dataset (different device, different
      task, subjects never seen), with PER-SUBJECT (per-user) normalization applied
      to each MAUS subject first (the deployment-time calibration).

MAUS labeling matches the training contrast: true REST vs working-memory LOAD
(2-back + 3-back). The ambiguous 0-back (a task with ~no working-memory load) is
excluded, since it is neither rest nor load — including it as "rest" understates
transfer and does not match how the model was trained.

Run: python3 ml/focus/src/external_validation_maus.py
"""
import os, sys, glob, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'stress'))
from src.features import compute_time_domain, compute_nonlinear  # noqa
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FEATURES = ['meanRR','sdnn','rmssd','pnn50','pnn20','hrMean','hrStd','hrRange','cvRR',
            'sd1','sd2','sd1sd2Ratio','sampleEntropy','dfaAlpha1']
NBACK = {1: 0, 2: 2, 3: 3, 4: 2, 5: 3, 6: 0}   # trial number -> n-back level
PARAMS = dict(objective='binary:logistic', eval_metric='logloss', n_estimators=400,
              learning_rate=0.05, max_depth=2, subsample=0.8, colsample_bytree=0.8,
              reg_lambda=2.0, reg_alpha=0.5, gamma=0.2, random_state=42, n_jobs=-1)


def hrv(rr):
    rr = np.asarray(rr, float); rr = rr[(rr >= 300) & (rr <= 2000)]
    if len(rr) > 5:
        m = np.median(rr); rr = rr[np.abs(rr - m) / m < 0.30]
    if len(rr) < 20:
        return None
    d = {}; d.update(compute_time_domain(rr)); d.update(compute_nonlinear(rr))
    return {k: float(d.get(k, 0.0)) for k in FEATURES}


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
            if n.startswith('rest'):
                cond = 'rest'
            elif n.startswith('trial_'):
                cond = f'{NBACK[int(n.split("_")[1])]}back'
            else:
                continue
            try:
                d = pd.read_csv(f)
            except Exception:
                continue
            if 'RRI_inf' not in d.columns:
                continue
            fe = hrv(d['RRI_inf'].values)
            if fe:
                rows.append({'subject_id': f'M_{sub}', 'cond': cond, **fe})
    return pd.DataFrame(rows)


def per_subject_z(df):
    o = df.copy()
    for f in FEATURES:
        g = o.groupby('subject_id')[f]
        o[f] = (o[f] - g.transform('mean')) / g.transform('std').replace(0, 1.0).fillna(1.0)
    return o.fillna(0.0)


def main():
    cog = pd.read_csv(os.path.join(BASE, 'focus', 'data', 'cogwear_features_samsung.csv'))
    cog = per_subject_z(cog)
    maus = load_maus()
    maus = maus[maus.cond.isin(['rest', '2back', '3back'])].copy()   # training-matched contrast
    maus['label'] = maus.cond.isin(['2back', '3back']).astype(int)
    maus = per_subject_z(maus)

    Xtr, ytr = cog[FEATURES].values, cog['label'].values
    pos = max(1, (ytr == 1).sum()); neg = max(1, (ytr == 0).sum())
    model = xgb.XGBClassifier(**PARAMS, scale_pos_weight=neg / pos)
    model.fit(Xtr, ytr)

    X, y = maus[FEATURES].values, maus['label'].values
    p = model.predict_proba(X)[:, 1]; yp = (p >= 0.5).astype(int)
    base = max((y == 0).mean(), (y == 1).mean())
    print("Deployed model: trained on CogWear only (rest vs Stroop), 20 subjects")
    print(f"External test on MAUS (rest vs 2/3-back), per-user normalized, "
          f"{maus.subject_id.nunique()} subjects, {len(maus)} windows")
    print(f"  AUC={roc_auc_score(y,p):.3f}  acc={accuracy_score(y,yp):.3f}  "
          f"F1={f1_score(y,yp):.3f}  MCC={matthews_corrcoef(y,yp):.3f}   [baseline acc={base:.3f}]")


if __name__ == '__main__':
    main()
