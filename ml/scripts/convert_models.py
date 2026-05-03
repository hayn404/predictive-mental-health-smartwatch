"""
Convert both models to the TypeScript-compatible custom JSON format.

Fixes:
  1. stress_model.json  — trees use f{i} indices, not feature names → inference always
                          takes "missing" path → constant output of ~38. Fix by replacing
                          f{i} with actual feature names throughout every tree node.

  2. anxiety_model.json — native XGBoost format; TypeScript engine can't read it at all
                          → falls back to rule-based. Fix by exporting via get_dump()
                          (same as stress model pipeline), injecting normalization params
                          computed from the training CSV, and adding required metadata.
"""

import json
import re
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE = Path(__file__).resolve().parent.parent   # ml/
MODELS = BASE / 'models'
DATA   = BASE / 'data'

STRESS_IN   = MODELS / 'stress_model.json'
ANXIETY_IN  = MODELS / 'anxiety_model.json'
STRESS_OUT  = MODELS / 'stress_model.json'          # overwrite in-place
ANXIETY_OUT = MODELS / 'anxiety_model_ts.json'      # new TS-compatible file

ASSETS_ML = BASE.parent / 'assets' / 'ml'
ASSETS_ML.mkdir(parents=True, exist_ok=True)

# ─── Feature order (must match training) ─────────────────────────────────────

FEATURES = [
    'meanRR', 'sdnn', 'rmssd', 'pnn50', 'pnn20',
    'hrMean', 'hrStd', 'hrRange', 'cvRR',
    'vlfPower', 'lfPower', 'hfPower', 'lfHfRatio',
    'totalPower', 'lfNorm', 'hfNorm',
    'sd1', 'sd2', 'sd1sd2Ratio', 'sampleEntropy', 'dfaAlpha1',
    'tempMean', 'tempSlope', 'tempStd', 'tempRange',
    'accelMagnitudeMean', 'accelMagnitudeStd', 'stepCount', 'activityType',
]

IDX_TO_NAME = {i: name for i, name in enumerate(FEATURES)}

# ─── Utility: replace f{i} split keys with feature names recursively ─────────

def fix_splits(node: dict) -> dict:
    """Walk a dump-format tree node and replace 'f5' → 'hrMean' etc."""
    if 'split' in node:
        m = re.fullmatch(r'f(\d+)', node['split'])
        if m:
            node['split'] = IDX_TO_NAME[int(m.group(1))]
    for child in node.get('children', []):
        fix_splits(child)
    return node

# ─── 1. Fix stress_model.json ─────────────────────────────────────────────────

print('=== Fixing stress_model.json ===')
with open(STRESS_IN) as f:
    sm = json.load(f)

before_split = sm['trees'][0].get('split')
fixed_trees = [fix_splits(tree) for tree in sm['trees']]
after_split = fixed_trees[0].get('split')
print(f'  Tree[0] split: {before_split!r} → {after_split!r}')

sm['trees'] = fixed_trees
with open(STRESS_OUT, 'w') as f:
    json.dump(sm, f, separators=(',', ':'))   # compact — keeps file small
import shutil
shutil.copy(STRESS_OUT, ASSETS_ML / 'stress_model.json')
print(f'  Saved {STRESS_OUT} ({STRESS_OUT.stat().st_size / 1024:.0f} KB)')
print(f'  Copied → assets/ml/stress_model.json')

# ─── 2. Convert anxiety_model.json → anxiety_model_ts.json ───────────────────

print('\n=== Converting anxiety_model.json ===')

# 2a. Load the booster
booster = xgb.Booster()
booster.load_model(str(ANXIETY_IN))
print(f'  Booster loaded — num_trees={booster.num_boosted_rounds()}')

# 2b. Get tree dump in nested JSON (same structure as stress model)
raw_dumps = booster.get_dump(dump_format='json')
trees = [fix_splits(json.loads(t)) for t in raw_dumps]
print(f'  Exported {len(trees)} trees with named splits')
print(f'  Tree[0] split: {trees[0]["split"]!r}')

# 2c. Compute normalization from training CSV
csv_path = DATA / 'anxiety_training_data.csv'
df = pd.read_csv(csv_path)
act_map = {'sedentary': 0, 'walking': 1, 'active': 2, 'sleeping': 3}
df['activityType'] = df['activityType'].map(act_map).fillna(0).astype(float)

X = df[FEATURES].copy()
scaler = StandardScaler()
scaler.fit(X)
norm_mean = dict(zip(FEATURES, scaler.mean_.tolist()))
norm_std  = dict(zip(FEATURES, scaler.scale_.tolist()))
print(f'  Normalization computed from {len(df)} training windows')
print(f'  rmssd  mean={norm_mean["rmssd"]:.2f}  std={norm_std["rmssd"]:.2f}')
print(f'  hrMean mean={norm_mean["hrMean"]:.2f}  std={norm_std["hrMean"]:.2f}')

# 2d. Base score from native model metadata
with open(ANXIETY_IN) as f:
    native = json.load(f)
    bs_val = native['learner']['learner_model_param']['base_score']
    if isinstance(bs_val, str) and bs_val.startswith('['):
        bs_val = bs_val.strip('[]')
    base_score = float(bs_val)
print(f'  Base score: {base_score:.4f}')

# 2e. Feature importances (weight = # times feature used for splitting)
raw_imp = booster.get_score(importance_type='weight')
total_w = sum(raw_imp.values()) or 1
importances = {}
for key, val in raw_imp.items():
    m = re.fullmatch(r'f(\d+)', key)
    name = IDX_TO_NAME[int(m.group(1))] if m else key
    importances[name] = round(val / total_w, 6)

top5 = sorted(importances.items(), key=lambda x: -x[1])[:5]
print(f'  Top 5 features: {top5}')

# 2f. Assemble export dict
export = {
    'version':      '1.0.0',
    'modelType':    'xgboost_regressor',
    'task':         'anxiety_prediction',
    'features':     FEATURES,
    'numFeatures':  len(FEATURES),
    'numTrees':     len(trees),
    'baseScore':    base_score,
    'learningRate': 0.01,
    'normalization': {
        'mean': norm_mean,
        'std':  norm_std,
    },
    'trees': trees,
    'metrics': {
        'mae':                     10.64,
        'rmse':                    12.88,
        'classification_accuracy': 0.88,
        'precision_severe':        0.96,
        'recall_severe':           0.86,
        'f1_severe':               0.91,
    },
    'importances': importances,
    'anxietyLevels': {
        'minimal':  {'min':  0, 'max': 20, 'label': 'Minimal',  'color': '#35e27e'},
        'mild':     {'min': 21, 'max': 45, 'label': 'Mild',     'color': '#FFB84D'},
        'moderate': {'min': 46, 'max': 70, 'label': 'Moderate', 'color': '#FF8A3D'},
        'severe':   {'min': 71, 'max': 100,'label': 'Severe',   'color': '#FF4D4D'},
    },
}

with open(ANXIETY_OUT, 'w') as f:
    json.dump(export, f, separators=(',', ':'))
shutil.copy(ANXIETY_OUT, ASSETS_ML / 'anxiety_model.json')
print(f'  Saved {ANXIETY_OUT} ({ANXIETY_OUT.stat().st_size / 1024:.0f} KB)')
print(f'  Copied → assets/ml/anxiety_model.json')

print('\n✅ Both models converted successfully.')
