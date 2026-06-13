import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# --- Paths ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'anxiety_training_data.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'anxiety_model.json')

# Re-define exact feature order to match on-device inference
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
    print("--- Seren Anxiety Model Training Pipeline ---")
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found!")
        return

    # 1. Load Data
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows.")

    # 2. Extract Features, Target, and Groups (Subjects)
    X = df[FEATURE_ORDER].values
    y = df['anxiety_index'].values
    groups = df['subject_id'].values

    # 3. Model Architecture (XGBoost Regressor)
    # Hyperparameters tuned for small/medium dataset to prevent overfitting
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=5000,       # Massive limit, early stopping will cut it off
        learning_rate=0.01,      # Extremely slow learning rate for maximum precision
        max_depth=6,             
        subsample=0.7,           
        colsample_bytree=0.7,
        reg_lambda=2.0,          
        reg_alpha=0.5,           
        gamma=0.2,               
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50
    )

    # 4. Leave-One-Subject-Out Cross Validation (LOSO-CV)
    print("\nStarting Leave-One-Subject-Out Cross-Validation (LOSO-CV)...")
    logo = LeaveOneGroupOut()
    
    best_iters = []
    maes = []
    rmses = []
    fold = 1
    
    unique_subjects = np.unique(groups)
    print(f"Evaluating across {len(unique_subjects)} unique subjects...")
    
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Fit with Early Stopping!
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Track the best iteration found before it overfit
        best_iters.append(model.best_iteration)
        
        preds = model.predict(X_test)
        
        # Ensure predictions are bounded between 0 and 100 just in case
        preds = np.clip(preds, 0, 100)
        
        mae = mean_absolute_error(y_test, preds)
        rmse = root_mean_squared_error(y_test, preds)
        
        maes.append(mae)
        rmses.append(rmse)
        fold += 1

    mean_mae = np.mean(maes)
    mean_rmse = np.mean(rmses)
    print("\n--- Cross-Validation Results ---")
    print(f"Mean Absolute Error (MAE): {mean_mae:.2f} points (out of 100)")
    print(f"Root Mean Squared Error (RMSE): {mean_rmse:.2f} points")
    print("--------------------------------")
    
    if mean_mae < 15:
        print("✅ Excellent MAE! The model predicts anxiety reliably within a reasonable margin.")
    elif mean_mae < 25:
        print("⚠️ Acceptable MAE, but could be tighter.")
    else:
        print("❌ High MAE. Model struggles to generalize.")

    avg_best_iter = int(np.mean(best_iters))
    print(f"Optimal Tree Count found via Early Stopping: {avg_best_iter} trees")

    # 5. Train Final Model on ALL Data
    print("\nTraining final model on 100% of data using optimal trees...")
    final_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=avg_best_iter, # EXACT optimal trees found to prevent overfitting
        learning_rate=0.01,
        max_depth=6,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=2.0,
        reg_alpha=0.5,
        gamma=0.2,
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(X, y)

    # 6. Extract Feature Importance
    importance = final_model.feature_importances_
    # Sort and print top 10
    feature_importance_tuples = sorted(zip(FEATURE_ORDER, importance), key=lambda x: x[1], reverse=True)
    print("\n--- Top 10 Most Important Features ---")
    for feat, imp in feature_importance_tuples[:10]:
        print(f"  {feat}: {imp*100:.1f}% contribution")

    # 7. Export Model for Device Inference
    os.makedirs(MODEL_DIR, exist_ok=True)
    final_model.save_model(MODEL_PATH)
    print(f"\n🚀 Success! Model weights exported to: {MODEL_PATH}")
    print("This JSON file is ready to be loaded by the TypeScript engine on the smartwatch!")

if __name__ == "__main__":
    train_and_evaluate()
