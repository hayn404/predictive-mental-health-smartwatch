"""
Biological-age model: HRV → age regression.
GLOBAL normalization (age is between-subject; per-subject norm would erase the signal).
Window-level training, subject-level evaluation (mean predicted age per subject).

LOSO on the training set + external test on the other dataset.
Run: python3 ml/bioage/src/train_bioage.py
"""
import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
import xgboost as xgb
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, r2_score

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FEAT = ['meanRR','sdnn','rmssd','pnn50','pnn20','hrMean','cvRR','sd1','sd2','sd1sd2Ratio','sampleEntropy','dfaAlpha1']
PARAMS = dict(objective='reg:squarederror', n_estimators=400, learning_rate=0.03, max_depth=3,
              subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, reg_alpha=0.5, random_state=42, n_jobs=-1)


def gnorm(X, mean=None, std=None):
    if mean is None:
        mean, std = X.mean(0), X.std(0) + 1e-9
    return (X - mean) / std, mean, std


def subj_level(y_true, y_pred, groups):
    df = pd.DataFrame({'g': groups, 'yt': y_true, 'yp': y_pred})
    a = df.groupby('g').agg(yt=('yt', 'first'), yp=('yp', 'mean'))
    return mean_absolute_error(a.yt, a.yp), r2_score(a.yt, a.yp), np.corrcoef(a.yt, a.yp)[0, 1]


def loso(df, label):
    X = df[FEAT].values; y = df['age'].values.astype(float); g = df['subject_id'].values
    logo = LeaveOneGroupOut(); yt, yp, gg = [], [], []
    for tr, te in logo.split(X, y, g):
        Xtr, m, s = gnorm(X[tr]); Xte, _, _ = gnorm(X[te], m, s)
        reg = xgb.XGBRegressor(**PARAMS); reg.fit(Xtr, y[tr])
        yp.extend(reg.predict(Xte)); yt.extend(y[te]); gg.extend(g[te])
    mae, r2, r = subj_level(np.array(yt), np.array(yp), np.array(gg))
    print(f"{label:32s} subject-level: MAE={mae:.2f} yrs  r={r:.3f}  R2={r2:.3f}")


def external(train_df, test_df, label):
    Xtr, m, s = gnorm(train_df[FEAT].values); ytr = train_df['age'].values.astype(float)
    reg = xgb.XGBRegressor(**PARAMS); reg.fit(Xtr, ytr)
    Xte, _, _ = gnorm(test_df[FEAT].values, m, s)
    yp = reg.predict(Xte)
    mae, r2, r = subj_level(test_df['age'].values.astype(float), yp, test_df['subject_id'].values)
    print(f"{label:32s} subject-level: MAE={mae:.2f} yrs  r={r:.3f}  R2={r2:.3f}")


def main():
    fan = os.path.join(BASE, 'data', 'bioage_features_fantasia.csv')
    aa = os.path.join(BASE, 'data', 'bioage_features_aa.csv')
    df_fan = pd.read_csv(fan).replace([np.inf, -np.inf], np.nan).fillna(0.0) if os.path.exists(fan) else None
    df_aa = pd.read_csv(aa).replace([np.inf, -np.inf], np.nan).fillna(0.0) if os.path.exists(aa) else None

    if df_fan is not None:
        print(f"Fantasia: {len(df_fan)} win, {df_fan.subject_id.nunique()} subj, age {df_fan.age.min():.0f}-{df_fan.age.max():.0f}")
        loso(df_fan, "Fantasia LOSO")
    if df_aa is not None:
        print(f"AA: {len(df_aa)} win, {df_aa.subject_id.nunique()} subj")
        loso(df_aa, "Autonomic Aging LOSO")
    if df_aa is not None and df_fan is not None:
        external(df_aa, df_fan, "Train AA → Test Fantasia")


if __name__ == '__main__':
    main()
