"""
Export the trained bio-age XGBoost regressor to the TypeScript tree format the app runs
(same shape as the focus/stress models). Regression → prediction = baseScore + sum(leaves)
(NO sigmoid). Carries the GLOBAL normalization stats (bio-age is between-subject).

Output: assets/ml/bioage/bioage_model.json
"""
import os, json, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
import xgboost as xgb

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))   # repo root
MDIR = os.path.join(BASE, 'ml', 'bioage', 'models')
meta = json.load(open(os.path.join(MDIR, 'bioage_model_meta.json')))
FEAT = meta['features']

reg = xgb.XGBRegressor()
reg.load_model(os.path.join(MDIR, 'bioage_model.json'))
booster = reg.get_booster()

idx_to_name = {i: n for i, n in enumerate(FEAT)}


def fix(node):
    if 'split' in node and isinstance(node['split'], str) and node['split'].startswith('f'):
        try:
            node['split'] = idx_to_name[int(node['split'][1:])]
        except (ValueError, KeyError):
            pass
    for c in node.get('children', []):
        fix(c)
    return node


trees = [fix(json.loads(d)) for d in booster.get_dump(dump_format='json')]
cfg = json.loads(booster.save_config())
base = float(cfg['learner']['learner_model_param']['base_score'])

model = {
    'version': 'bioage-aa-1.0',
    'modelType': 'xgboost',
    'task': 'regression',
    'applySigmoid': False,
    'features': FEAT,
    'baseScore': base,
    'normalization': {
        'mean': {f: float(meta['norm_mean'][i]) for i, f in enumerate(FEAT)},
        'std':  {f: float(meta['norm_std'][i]) for i, f in enumerate(FEAT)},
    },
    'representation': 'subject-aggregate (median of recent HRV windows)',
    'trees': trees,
    'metrics': {'loso_mae_years': meta['loso_mae'], 'loso_r': meta['loso_r'],
                'cross_dataset_young_old_auc': 0.94},
    'trainingNotes': ('Trained on PhysioNet Autonomic Aging (1095 subjects). Predicts physiological '
                      'age from 14 HRV features; GLOBAL normalization; output age-gap = predicted - chronological. '
                      'Aggregate the user recent windows (median) before predicting.'),
}
out = os.path.join(BASE, 'assets', 'ml', 'bioage', 'bioage_model.json')
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(model, open(out, 'w'))
print(f"exported -> {os.path.relpath(out, BASE)}  ({len(trees)} trees, baseScore={base:.2f})")

# ---- verify TS-format reproduces the booster ----
fan = pd.read_csv(os.path.join(BASE, 'ml', 'bioage', 'data', 'bioage_features_fantasia.csv')).replace([np.inf, -np.inf], np.nan).fillna(0.0)
ag = fan.groupby('subject_id')[FEAT].median()


def trav(n, f):
    if 'leaf' in n:
        return n['leaf']
    v = f.get(n['split'])
    nid = n['no'] if (v is None or np.isnan(v)) else (n['yes'] if v < n['split_condition'] else n['no'])
    for c in n['children']:
        if c['nodeid'] == nid:
            return trav(c, f)
    return 0.0


m = model['normalization']['mean']; s = model['normalization']['std']
mine, xgbp = [], []
Xn = ((ag[FEAT] - np.array(meta['norm_mean'])) / np.array(meta['norm_std']))
xgb_pred = reg.predict(Xn.values)
for k, (_, row) in enumerate(ag.iterrows()):
    fd = {f: (row[f] - m[f]) / s[f] for f in FEAT}
    p = base + sum(trav(t, fd) for t in trees)
    mine.append(p); xgbp.append(xgb_pred[k])
print(f"TS vs XGBoost max abs diff: {np.max(np.abs(np.array(mine) - np.array(xgbp))):.4f} (should be ~0)")
