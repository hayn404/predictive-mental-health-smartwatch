import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'academic_training_data.csv')
MODEL_DIR  = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'focus_model.json')

FEATURE_ORDER = [
    'meanRR', 'sdnn', 'rmssd', 'pnn50', 'pnn20',
    'hrMean', 'hrStd', 'hrRange', 'cvRR',
    'vlfPower', 'lfPower', 'hfPower', 'lfHfRatio',
    'totalPower', 'lfNorm', 'hfNorm',
    'sd1', 'sd2', 'sd1sd2Ratio', 'sampleEntropy', 'dfaAlpha1',
    'tempMean', 'tempSlope', 'tempStd', 'tempRange',
    'accelMagnitudeMean', 'accelMagnitudeStd', 'stepCount', 'activityType',
]


def train_and_evaluate():
    print('--- Seren Academic Performance Model Training Pipeline ---')

    if not os.path.exists(DATA_PATH):
        print(f'Data not found: {DATA_PATH}')
        print('Run extract_exam_features.py first.')
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    print(f'Loaded {len(df)} windows from {df["subject_id"].nunique()} subjects, '
          f'{df["session_name"].nunique()} session types.')
    print(f'Grade range: {df["grade"].min():.1f} – {df["grade"].max():.1f}')

    X      = df[FEATURE_ORDER].values
    y      = df['grade'].values
    groups = df['subject_id'].values

    model_params = dict(
        objective='reg:squarederror',
        n_estimators=5000,
        learning_rate=0.01,
        max_depth=4,          # shallower than anxiety — only 10 subjects, risk of overfit
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=3.0,       # stronger regularisation for small N
        reg_alpha=0.5,
        gamma=0.3,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
    )

    print('\nStarting Leave-One-Subject-Out Cross-Validation (LOSO-CV)...')
    logo = LeaveOneGroupOut()

    best_iters = []
    maes  = []
    rmses = []

    unique_subjects = np.unique(groups)
    print(f'Evaluating across {len(unique_subjects)} subjects...')

    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        fold_model = xgb.XGBRegressor(**model_params)
        fold_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        best_iters.append(fold_model.best_iteration)
        preds = np.clip(fold_model.predict(X_test), 0, 100)

        maes.append(mean_absolute_error(y_test, preds))
        rmses.append(root_mean_squared_error(y_test, preds))

    mean_mae  = np.mean(maes)
    mean_rmse = np.mean(rmses)
    print('\n--- Cross-Validation Results ---')
    print(f'Mean Absolute Error  (MAE):  {mean_mae:.2f} grade points')
    print(f'Root Mean Sq Error   (RMSE): {mean_rmse:.2f} grade points')
    print('--------------------------------')

    if mean_mae < 10:
        print('Excellent: model predicts grade within ±10 points.')
    elif mean_mae < 20:
        print('Acceptable: model predicts grade within ±20 points.')
    else:
        print('High error — dataset is very small (N=10), consider ensemble or transfer learning.')

    avg_best_iter = int(np.mean(best_iters))
    print(f'Optimal tree count (early stopping): {avg_best_iter}')

    # Final model on all data
    print('\nTraining final model on 100% of data...')
    final_params = {k: v for k, v in model_params.items() if k != 'early_stopping_rounds'}
    final_params['n_estimators'] = avg_best_iter

    final_model = xgb.XGBRegressor(**final_params)
    final_model.fit(X, y)

    importance = final_model.feature_importances_
    ranked = sorted(zip(FEATURE_ORDER, importance), key=lambda x: x[1], reverse=True)
    print('\n--- Top 10 Most Important Features ---')
    for feat, imp in ranked[:10]:
        print(f'  {feat}: {imp * 100:.1f}%')

    os.makedirs(MODEL_DIR, exist_ok=True)
    final_model.save_model(MODEL_PATH)
    print(f'\nModel saved to: {MODEL_PATH}')


if __name__ == '__main__':
    train_and_evaluate()
