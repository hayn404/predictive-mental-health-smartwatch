"""
SHAP explainability (XAI) for the focus / engagement model.

Trains the final XGBoost model on CogWear (per-subject normalized) using the PSO-tuned
hyperparameters (falls back to shipped params), then computes SHAP values to show:
  - which features drive the prediction (mean |SHAP|)
  - the DIRECTION of each feature (does a high value push toward "engaged" or "rest")

Outputs a ranked table + saves a beeswarm summary plot if matplotlib is available.

Run: python3 ml/focus/src/shap_xai.py
"""
import os, sys, json, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
import xgboost as xgb

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FEAT = ['meanRR','sdnn','rmssd','pnn50','pnn20','hrMean','hrStd','hrRange','cvRR',
        'sd1','sd2','sd1sd2Ratio','sampleEntropy','dfaAlpha1']


def per_subject_z(df):
    o = df.copy()
    for f in FEAT:
        g = o.groupby('subject_id')[f]
        o[f] = (o[f] - g.transform('mean')) / g.transform('std').replace(0, 1.0).fillna(1.0)
    return o.fillna(0.0)


def main():
    try:
        import shap
    except ImportError:
        print("shap not installed. Run: pip3 install --user shap")
        sys.exit(1)

    cog = per_subject_z(pd.read_csv(os.path.join(BASE, 'focus', 'data', 'cogwear_features_samsung.csv')))
    X = cog[FEAT]; y = cog['label'].values.astype(int)

    pf = os.path.join(BASE, 'focus', 'models', 'pso_best_params.json')
    if os.path.exists(pf):
        params = json.load(open(pf))['best_params']
        print("using PSO-tuned params")
    else:
        params = dict(max_depth=2, learning_rate=0.05, n_estimators=400, subsample=0.8,
                      colsample_bytree=0.8, reg_lambda=2.0, reg_alpha=0.5, gamma=0.2, min_child_weight=1)
        print("using shipped params (no PSO file found)")

    pos = max(1, (y == 1).sum()); neg = max(1, (y == 0).sum())
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                              scale_pos_weight=neg / pos, random_state=42, n_jobs=-1, **params)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)               # (n, 14) in margin (logit) space
    mean_abs = np.abs(sv).mean(axis=0)
    # direction: sign of correlation between feature value and its SHAP contribution
    direction = []
    for i, f in enumerate(FEAT):
        c = np.corrcoef(X[f].values, sv[:, i])[0, 1]
        direction.append('high → engaged' if c > 0 else 'high → rest')

    order = np.argsort(mean_abs)[::-1]
    print(f"\n{'feature':16s}{'mean|SHAP|':>12s}   direction")
    print("-" * 48)
    rows = []
    for i in order:
        print(f"{FEAT[i]:16s}{mean_abs[i]:>12.4f}   {direction[i]}")
        rows.append({'feature': FEAT[i], 'mean_abs_shap': float(mean_abs[i]), 'direction': direction[i]})

    json.dump(rows, open(os.path.join(BASE, 'focus', 'models', 'shap_importance.json'), 'w'), indent=2)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        shap.summary_plot(sv, X, show=False, max_display=14)
        out = os.path.join(BASE, 'focus', 'models', 'shap_summary.png')
        plt.tight_layout(); plt.savefig(out, dpi=140, bbox_inches='tight'); plt.close()
        print(f"\nsaved beeswarm plot -> {out}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == '__main__':
    main()
