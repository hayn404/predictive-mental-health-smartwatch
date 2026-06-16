"""
Train + export the Seren focus / cognitive-effort model on CogWear HRV features.

Task: binary classification  rest (0) vs cognitive effort (1).
Method: PER-USER normalization (z-score each feature within the subject) + XGBoost.
Validation: Leave-One-Subject-Out CV.

Exports directly to the TypeScript tree format that services/ai/focusModel.ts consumes
(same shape as assets/ml/stress/stress_model.json), so the watch runs the real model.

Run:  python3 ml/focus/src/train_cogwear_focus.py samsung
"""

import os
import sys
import json
import math
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEVICE = sys.argv[1] if len(sys.argv) > 1 else 'samsung'
CSV = os.path.join(BASE, 'focus', 'data', f'cogwear_features_{DEVICE}.csv')

# 14 HRV features the watch reliably reproduces from Health Connect HR (featureEngineering.ts).
FEATURES = ['meanRR', 'sdnn', 'rmssd', 'pnn50', 'pnn20', 'hrMean', 'hrStd', 'hrRange', 'cvRR',
            'sd1', 'sd2', 'sd1sd2Ratio', 'sampleEntropy', 'dfaAlpha1']

PARAMS = dict(objective='binary:logistic', eval_metric='logloss',
              n_estimators=400, learning_rate=0.05, max_depth=2,
              subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, reg_alpha=0.5,
              gamma=0.2, random_state=42, n_jobs=-1)


def per_user_normalize(df, feats):
    out = df.copy()
    for f in feats:
        g = out.groupby('subject_id')[f]
        out[f] = (out[f] - g.transform('mean')) / g.transform('std').replace(0, 1.0).fillna(1.0)
    return out.fillna(0.0)


def loso_eval(X, y, groups):
    logo = LeaveOneGroupOut(); yt, yp, ypr = [], [], []
    for tr, te in logo.split(X, y, groups):
        pos = max(1, (y[tr] == 1).sum()); neg = max(1, (y[tr] == 0).sum())
        m = xgb.XGBClassifier(**PARAMS, scale_pos_weight=neg / pos)
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        yt.extend(y[te]); ypr.extend(p); yp.extend((p >= 0.5).astype(int))
    yt, yp, ypr = map(np.array, (yt, yp, ypr))
    return (accuracy_score(yt, yp), f1_score(yt, yp),
            matthews_corrcoef(yt, yp), roc_auc_score(yt, ypr))


def fix_splits(node, idx_to_name):
    """get_dump json uses 'f{i}' split keys -> replace with feature names."""
    if 'split' in node and isinstance(node['split'], str) and node['split'].startswith('f'):
        try:
            node['split'] = idx_to_name[int(node['split'][1:])]
        except (ValueError, KeyError):
            pass
    for child in node.get('children', []):
        fix_splits(child, idx_to_name)
    return node


def export_ts(final, raw_df, metrics):
    booster = final.get_booster()
    dumps = booster.get_dump(dump_format='json')
    idx_to_name = {i: n for i, n in enumerate(FEATURES)}
    trees = [fix_splits(json.loads(d), idx_to_name) for d in dumps]

    cfg = json.loads(booster.save_config())
    base_prob = float(cfg['learner']['learner_model_param']['base_score'])
    base_prob = min(max(base_prob, 1e-6), 1 - 1e-6)
    base_margin = math.log(base_prob / (1 - base_prob))  # logit base for sigmoid

    # Global (population) mean/std on RAW features — cold-start fallback before the
    # user has enough personal history for per-user normalization.
    mean = {f: float(raw_df[f].mean()) for f in FEATURES}
    std = {f: float(raw_df[f].std() or 1.0) for f in FEATURES}
    importances = {f: float(v) for f, v in zip(FEATURES, final.feature_importances_)}

    model = {
        'version': f'cogwear-{DEVICE}-1.0',
        'modelType': 'xgboost',
        'task': 'binary_logistic',
        'features': FEATURES,
        'numFeatures': len(FEATURES),
        'numTrees': len(trees),
        'baseScore': base_margin,
        'applySigmoid': True,
        'learningRate': PARAMS['learning_rate'],
        'normalization': {'mean': mean, 'std': std},
        'perUserNorm': True,
        'decisionThreshold': 0.5,
        'trees': trees,
        'importances': importances,
        'metrics': {'loso_acc': metrics[0], 'loso_f1': metrics[1],
                    'loso_mcc': metrics[2], 'loso_auc': metrics[3]},
        'trainingNotes': ('Trained on CogWear (rest vs Stroop cognitive effort), Galaxy Watch4 PPG. '
                          'Per-user z-normalization at inference; global mean/std for cold-start. '
                          'Higher score = more cognitive effort/engagement detected.'),
    }
    out_assets = os.path.join(BASE, '..', 'assets', 'ml', 'focus', 'focus_model.json')
    os.makedirs(os.path.dirname(out_assets), exist_ok=True)
    with open(out_assets, 'w') as f:
        json.dump(model, f)
    print(f'exported TS model -> {os.path.relpath(out_assets, os.path.join(BASE, ".."))}')


def main():
    df = pd.read_csv(CSV).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feats = [f for f in FEATURES if f in df.columns]
    y = df['label'].to_numpy(int); groups = df['subject_id'].to_numpy()
    print(f'=== CogWear focus model [{DEVICE}] ===')
    print(f'windows={len(df)} subjects={df["subject_id"].nunique()} features={len(feats)}')
    maj = max((y == 0).mean(), (y == 1).mean())
    print(f'BASELINE (majority) acc={maj:.3f} MCC=0')

    dn = per_user_normalize(df, feats)
    X = dn[feats].to_numpy(float)
    metrics = loso_eval(X, y, groups)
    print(f'PER-USER model     acc={metrics[0]:.3f} F1={metrics[1]:.3f} MCC={metrics[2]:.3f} AUC={metrics[3]:.3f}')

    pos = max(1, (y == 1).sum()); neg = max(1, (y == 0).sum())
    final = xgb.XGBClassifier(**PARAMS, scale_pos_weight=neg / pos)
    final.fit(X, y)
    export_ts(final, df, metrics)


if __name__ == '__main__':
    main()
