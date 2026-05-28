#!/usr/bin/env python3
"""
Train final binary Stacking Ensemble (Home Win vs Not Home Win)

Saves model, test predictions, and a small results summary to `data/`.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
import joblib

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

EXCLUDE_COLS = {
    'game_id', 'home_club_id', 'away_club_id', 'date', 'competition_id',
    'home_feature_date', 'away_feature_date', 'season', 'round',
    'true_result', 'outcome', 'true_home_goals', 'true_away_goals',
    'home_club_goals', 'away_club_goals', 'target', 'home_club_name',
    'away_club_name', 'home_indicator'
}


def load_features(data_dir: Path):
    files = list(data_dir.glob('match_features*.csv'))
    if not files:
        print('ERROR: no match_features CSV found in data/')
        return None, None, None
    files.sort(key=lambda x: x.stat().st_size, reverse=True)
    df = pd.read_csv(files[0])
    # Binary label: Home Win (1) vs Not Home Win (0)
    if 'home_club_goals' in df.columns and 'away_club_goals' in df.columns:
        y = np.where(df['home_club_goals'] > df['away_club_goals'], 1, 0)
    elif 'true_result' in df.columns:
        y = np.where(df['true_result'] == 1, 1, 0)
    else:
        print('ERROR: cannot compute binary outcome')
        return None, None, None

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_cols].copy()
    return X, y, feature_cols


def train_and_save(data_dir: Path, models_dir: Path, output_dir: Path):
    X, y, _ = load_features(data_dir)
    if X is None:
        return

    n = len(X)
    n_test = int(0.20 * n)

    X_train = X.iloc[:-n_test]
    X_test = X.iloc[-n_test:]
    y_train = y[:-n_test]
    y_test = y[-n_test:]

    # Base learners
    estimators = []
    if HAS_XGBOOST:
        estimators.append(('xgb', XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                                                subsample=0.8, colsample_bytree=0.8,
                                                objective='binary:logistic', random_state=42, n_jobs=-1)))

    estimators.append(('rf', RandomForestClassifier(n_estimators=200, max_depth=10,
                                                   class_weight='balanced', random_state=42, n_jobs=-1)))

    lr_pipe = Pipeline([('scaler', RobustScaler()), ('imp', SimpleImputer(strategy='median')),
                        ('lr', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))])
    estimators.append(('lr', lr_pipe))

    meta = LogisticRegression(class_weight='balanced', random_state=42)

    stack = StackingClassifier(estimators=estimators, final_estimator=meta,
                               cv=5, stack_method='predict_proba', n_jobs=-1)

    # Impute training/test
    imputer = SimpleImputer(strategy='median')
    X_tr_imp = imputer.fit_transform(X_train)
    X_te_imp = imputer.transform(X_test)

    print(f"Training stacking ensemble on {len(X_tr_imp):,} samples...")
    stack.fit(X_tr_imp, y_train)

    y_pred = stack.predict(X_te_imp)
    y_prob = stack.predict_proba(X_te_imp)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc*100:.2f}%")
    print('\nClassification report:')
    print(classification_report(y_test, y_pred, digits=4))

    # Save model and predictions
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(stack, models_dir / 'stacking_final.pkl')

    output_dir.mkdir(parents=True, exist_ok=True)
    preds_df = pd.DataFrame({
        'true_home_win': y_test,
        'pred_home_win': y_pred,
        'pred_prob_home_win': y_prob,
    })
    preds_df.to_csv(output_dir / 'stacking_final_test_predictions.csv', index=False)

    # Save summary
    results = {
        'model': 'stacking_final',
        'test_samples': int(len(y_test)),
        'accuracy': float(acc),
    }
    (data_dir / 'model_results.csv').parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([results]).to_csv(data_dir / 'model_results.csv', index=False)
    (data_dir / 'model_results_summary.json').write_text(json.dumps([results], indent=2))

    print(f"Saved model to {models_dir / 'stacking_final.pkl'}")
    print(f"Saved predictions to {output_dir / 'stacking_final_test_predictions.csv'}")
    print(f"Saved results summary to {data_dir / 'model_results_summary.json'}")


def main():
    base = Path(__file__).resolve().parent.parent
    data_dir = base / 'data'
    models_dir = base / 'models'
    preds_dir = data_dir / 'predictions'

    train_and_save(data_dir, models_dir, preds_dir)


if __name__ == '__main__':
    main()
