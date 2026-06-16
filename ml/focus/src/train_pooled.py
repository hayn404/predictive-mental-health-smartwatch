"""
Train + validate the pooled engagement model (CogWear + MAUS + CLARE).

Two honest validations:
  LOSO  — leave-one-subject-out across all 62 subjects (cross-subject generalization)
  LODO  — leave-one-dataset-out: train on 2 datasets, test on the 3rd
          (cross-dataset / cross-device / cross-task generalization — the real test)

Per-subject z-normalization is applied (bridges the 3 devices). Reports per-fold
and overall metrics vs the majority-class baseline.

Run: python3 ml/focus/src/train_pooled.py
"""
import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
import xgboost as xgb
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CSV = os.path.join(BASE, 'focus', 'data', 'pooled_features.csv')
FEATURES = ['meanRR','sdnn','rmssd','pnn50','pnn20','hrMean','hrStd','hrRange','cvRR',
            'sd1','sd2','sd1sd2Ratio','sampleEntropy','dfaAlpha1']
PARAMS = dict(objective='binary:logistic', eval_metric='logloss', n_estimators=400,
              learning_rate=0.05, max_depth=2, subsample=0.8, colsample_bytree=0.8,
              reg_lambda=2.0, reg_alpha=0.5, gamma=0.2, random_state=42, n_jobs=-1)


def per_subject_norm(df):
    out = df.copy()
    for f in FEATURES:
        g = out.groupby('subject_id')[f]
        out[f] = (out[f] - g.transform('mean')) / g.transform('std').replace(0, 1.0).fillna(1.0)
    return out.fillna(0.0)


def fit_eval(Xtr, ytr, Xte, yte):
    pos = max(1, (ytr == 1).sum()); neg = max(1, (ytr == 0).sum())
    m = xgb.XGBClassifier(**PARAMS, scale_pos_weight=neg / pos)
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    yp = (p >= 0.5).astype(int)
    auc = roc_auc_score(yte, p) if len(np.unique(yte)) > 1 else float('nan')
    return accuracy_score(yte, yp), f1_score(yte, yp, zero_division=0), matthews_corrcoef(yte, yp), auc


def main():
    df = pd.read_csv(CSV).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    dn = per_subject_norm(df)
    X = dn[FEATURES].to_numpy(float)
    y = dn['label'].to_numpy(int)
    subj = dn['subject_id'].to_numpy()
    dset = dn['dataset'].to_numpy()

    maj = max((y == 0).mean(), (y == 1).mean())
    print(f"pooled: {len(df)} windows, {df['subject_id'].nunique()} subjects, 3 datasets")
    print(f"baseline (majority): acc={maj:.3f}\n")

    # ---- LOSO across all subjects ----
    logo = LeaveOneGroupOut(); yt, yp, ypr = [], [], []
    for tr, te in logo.split(X, y, subj):
        pos = max(1, (y[tr] == 1).sum()); neg = max(1, (y[tr] == 0).sum())
        m = xgb.XGBClassifier(**PARAMS, scale_pos_weight=neg / pos); m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        yt.extend(y[te]); ypr.extend(p); yp.extend((p >= 0.5).astype(int))
    yt, yp, ypr = map(np.array, (yt, yp, ypr))
    print("LOSO (cross-subject, pooled):")
    print(f"  acc={accuracy_score(yt,yp):.3f}  F1={f1_score(yt,yp):.3f}  "
          f"MCC={matthews_corrcoef(yt,yp):.3f}  AUC={roc_auc_score(yt,ypr):.3f}\n")

    # ---- LODO: leave-one-dataset-out ----
    print("LODO (leave-one-dataset-out — the generalization test):")
    for held in ['cogwear', 'maus', 'clare']:
        tr = dset != held; te = dset == held
        acc, f1, mcc, auc = fit_eval(X[tr], y[tr], X[te], y[te])
        print(f"  train on others → test on {held:8s}: acc={acc:.3f}  F1={f1:.3f}  MCC={mcc:.3f}  AUC={auc:.3f}")


if __name__ == '__main__':
    main()
