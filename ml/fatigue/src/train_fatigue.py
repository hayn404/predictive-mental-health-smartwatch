"""
Train + validate the mental-fatigue model (mirrors the focus pipeline).

Reads standardized feature CSVs produced by the per-dataset adapters:
    ml/fatigue/data/fatigue_features_mefar.csv   (TRAIN, Empatica E4 wrist BVP -> HRV)
    ml/fatigue/data/fatigue_features_sae.csv     (EXTERNAL TEST, ECG -> HRV)
each with columns: subject_id, label, [14 HRV features]   (label: 0 = fresh, 1 = fatigued)

Validations (same as focus):
  LOSO     — leave-one-subject-out on MEFAR (cross-subject)
  EXTERNAL — train on all MEFAR, test zero-shot on SAE (cross-dataset generalization)

Per-subject z-normalization throughout (the focus/stress accuracy lever).

Run: python3 ml/fatigue/src/train_fatigue.py
"""
import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
import xgboost as xgb
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FEAT = ['meanRR','sdnn','rmssd','pnn50','pnn20','hrMean','cvRR',
        'sd1','sd2','sd1sd2Ratio','sampleEntropy','dfaAlpha1']  # 12 shared MEFAR↔SAE
PARAMS = dict(objective='binary:logistic', eval_metric='logloss', n_estimators=400,
              learning_rate=0.05, max_depth=2, subsample=0.8, colsample_bytree=0.8,
              reg_lambda=2.0, reg_alpha=0.5, gamma=0.2, random_state=42, n_jobs=-1)


def per_subject_z(df):
    o = df.copy()
    for f in FEAT:
        g = o.groupby('subject_id')[f]
        o[f] = (o[f] - g.transform('mean')) / g.transform('std').replace(0, 1.0).fillna(1.0)
    return o.fillna(0.0)


def metrics(y, p):
    yp = (p >= 0.5).astype(int)
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float('nan')
    return accuracy_score(y, yp), f1_score(y, yp, zero_division=0), matthews_corrcoef(y, yp), auc


def main():
    train_csv = os.path.join(BASE, 'data', 'fatigue_features_mefar.csv')
    ext_csv = os.path.join(BASE, 'data', 'fatigue_features_sae.csv')
    if not os.path.exists(train_csv):
        print(f"missing {train_csv} — run the MEFAR adapter first."); return

    tr = per_subject_z(pd.read_csv(train_csv).replace([np.inf, -np.inf], np.nan).fillna(0.0))
    X, y, g = tr[FEAT].values, tr['label'].values.astype(int), tr['subject_id'].values
    maj = max((y == 0).mean(), (y == 1).mean())
    print(f"MEFAR: {len(tr)} windows, {tr.subject_id.nunique()} subjects | baseline acc={maj:.3f}")

    # LOSO
    logo = LeaveOneGroupOut(); yt, yp = [], []
    for trn, te in logo.split(X, y, g):
        pos = max(1, (y[trn] == 1).sum()); neg = max(1, (y[trn] == 0).sum())
        m = xgb.XGBClassifier(**PARAMS, scale_pos_weight=neg / pos); m.fit(X[trn], y[trn])
        yp.extend(m.predict_proba(X[te])[:, 1]); yt.extend(y[te])
    a, f, mc, au = metrics(np.array(yt), np.array(yp))
    print(f"LOSO (cross-subject):   acc={a:.3f} F1={f:.3f} MCC={mc:.3f} AUC={au:.3f}")

    # External
    if os.path.exists(ext_csv):
        ex = per_subject_z(pd.read_csv(ext_csv).replace([np.inf, -np.inf], np.nan).fillna(0.0))
        pos = max(1, (y == 1).sum()); neg = max(1, (y == 0).sum())
        m = xgb.XGBClassifier(**PARAMS, scale_pos_weight=neg / pos); m.fit(X, y)
        p = m.predict_proba(ex[FEAT].values)[:, 1]
        a, f, mc, au = metrics(ex['label'].values.astype(int), p)
        print(f"EXTERNAL (MEFAR→SAE):   acc={a:.3f} F1={f:.3f} MCC={mc:.3f} AUC={au:.3f}  (n={len(ex)}, {ex.subject_id.nunique()} subj)")
    else:
        print(f"(external test {ext_csv} not present yet)")


if __name__ == '__main__':
    main()
