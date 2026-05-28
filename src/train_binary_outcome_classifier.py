#!/usr/bin/env python3
"""
BINARY OUTCOME CLASSIFICATION (Home Win vs Not Home Win)
Baseline Comparison Script

This script trains individual models (XGBoost, Random Forest, Logistic Regression)
on the binary task to compare with the Stacking Ensemble results.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

def load_data(data_dir):
    feature_files = list(data_dir.glob("match_features*.csv"))
    if not feature_files:
        return None, None, None
    feature_files.sort(key=lambda x: x.stat().st_size, reverse=True)
    df = pd.read_csv(feature_files[0])
    
    if 'home_club_goals' in df.columns and 'away_club_goals' in df.columns:
        y = np.where(df['home_club_goals'] > df['away_club_goals'], 1, 0)
    elif 'true_result' in df.columns:
        y = np.where(df['true_result'] == 1, 1, 0)
    else:
        return None, None, None
    
    exclude_cols = {
        'game_id', 'home_club_id', 'away_club_id', 'date', 'competition_id',
        'home_feature_date', 'away_feature_date', 'season', 'round',
        'true_result', 'outcome', 'true_home_goals', 'true_away_goals',
        'home_club_goals', 'away_club_goals', 'target', 'home_club_name', 
        'away_club_name', 'home_indicator'
    }
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols]
    
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    return X_imputed, y, feature_cols

def train_individual_models(X, y):
    n_test = int(0.20 * len(X))
    X_train, X_test = X[:-n_test], X[-n_test:]
    y_train, y_test = y[:-n_test], y[-n_test:]
    
    # Weights
    class_counts = np.bincount(y_train)
    class_weights = len(y_train) / (2 * class_counts)
    
    results = {}
    
    # 1. XGBoost (Balanced)
    if HAS_XGBOOST:
        xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, 
                           objective='binary:logistic', random_state=42, n_jobs=-1)
        xgb.fit(X_train, y_train, sample_weight=class_weights[y_train])
        y_pred = xgb.predict(X_test)
        results['XGBoost (Balanced)'] = {
            'train_acc': xgb.score(X_train, y_train),
            'acc': accuracy_score(y_test, y_pred), 
            'recall': classification_report(y_test, y_pred, output_dict=True)['1']['recall']
        }

    # 2. XGBoost (Unbalanced - "Previous Model")
    if HAS_XGBOOST:
        xgb_unb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, 
                               objective='binary:logistic', random_state=42, n_jobs=-1)
        xgb_unb.fit(X_train, y_train)
        y_pred_unb = xgb_unb.predict(X_test)
        results['XGBoost (Unbalanced)'] = {
            'train_acc': xgb_unb.score(X_train, y_train),
            'acc': accuracy_score(y_test, y_pred_unb), 
            'recall': classification_report(y_test, y_pred_unb, output_dict=True)['1']['recall']
        }

    # 3. Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    results['Random Forest'] = {
        'train_acc': rf.score(X_train, y_train),
        'acc': accuracy_score(y_test, y_pred), 
        'recall': classification_report(y_test, y_pred, output_dict=True)['1']['recall']
    }
    
    # 4. Logistic Regression
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_train_scaled, y_train)
    y_pred = lr.predict(X_test_scaled)
    results['Logistic Regression'] = {
        'train_acc': lr.score(X_train_scaled, y_train),
        'acc': accuracy_score(y_test, y_pred), 
        'recall': classification_report(y_test, y_pred, output_dict=True)['1']['recall']
    }
    
    return results

def main():
    base_dir = Path(__file__).resolve().parent.parent
    X, y, _ = load_data(base_dir / "data")
    if X is None: return
    
    print("\n" + "="*70)
    print("TRAINING INDIVIDUAL MODELS (Home Win vs Not Home Win)")
    print("="*70)
    
    results = train_individual_models(X, y)
    
    print("\n📊 Results Comparison:")
    print(f"{'Model':25s} | {'Train Acc':10s} | {'Test Acc':10s} | {'Home Win Recall':15s}")
    print("-" * 70)
    for model, metrics in results.items():
        print(f"{model:25s} | {metrics['train_acc']*100:8.2f}% | {metrics['acc']*100:8.2f}% | {metrics['recall']*100:14.2f}%")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
