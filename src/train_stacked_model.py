#!/usr/bin/env python3
"""
Implementation of a Stacking Ensemble model based on:
"A predictive analytics framework for forecasting soccer match outcomes using machine learning models"
Decision Analytics Journal, 2024 (PII S2772-6622(24)00141-3)

Models used: XGBoost, Random Forest, Logistic Regression.
Meta-learner: Logistic Regression.
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("WARNING: xgboost not installed")

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(data_dir):
    """Load features and outcomes from match_features.csv"""
    feature_files = list(data_dir.glob("match_features*.csv"))
    if not feature_files:
        print("ERROR: No match_features*.csv found in data/")
        return None, None, None
    
    # Prefer the larger file if multiple exist
    feature_files.sort(key=lambda x: x.stat().st_size, reverse=True)
    features_path = feature_files[0]
    print(f"Loading features from: {features_path.name}")
    
    df = pd.read_csv(features_path)
    print(f"✓ Loaded {len(df)} matches")
    
    # Compute outcome (1=Home, 0=Draw, -1=Away)
    if 'home_club_goals' in df.columns and 'away_club_goals' in df.columns:
        home_goals = df['home_club_goals'].values
        away_goals = df['away_club_goals'].values
        y = np.where(home_goals > away_goals, 1, 
                    np.where(home_goals == away_goals, 0, -1))
    elif 'true_result' in df.columns:
        y = df['true_result'].values
    else:
        print("ERROR: Cannot compute outcome")
        return None, None, None
    
    # Remap outcomes: 1→1 (Home Win), 0,-1→0 (Not Home Win)
    y_binary = np.where(y == 1, 1, 0)
    
    # Feature selection
    exclude_cols = {
        'game_id', 'home_club_id', 'away_club_id', 'date', 'competition_id',
        'home_feature_date', 'away_feature_date', 'season', 'round',
        'true_result', 'outcome', 'true_home_goals', 'true_away_goals',
        'home_club_goals', 'away_club_goals', 'target', 'home_club_name', 
        'away_club_name', 'home_indicator'
    }
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols]
    
    print(f"✓ Features: {len(feature_cols)}")
    print(f"  Outcome distribution: Home Win {(y_binary==1).sum()} | Not Home Win {(y_binary==0).sum()}")
    
    return X, y_binary, feature_cols

# ══════════════════════════════════════════════════════════════════════════════
# STACKING ENSEMBLE TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_stacked_model(X, y, feature_cols):
    """Implement and train the Stacking Ensemble model for BINARY classification"""
    
    print("\n" + "="*70)
    print("IMPLEMENTING BINARY STACKING ENSEMBLE (Home Win vs Not Home Win)")
    print("="*70)
    
    # Split data temporally (last 20% for test)
    n_total = len(X)
    n_test = int(0.20 * n_total)
    
    X_train = X.iloc[:-n_test]
    y_train = y[:-n_test]
    X_test = X.iloc[-n_test:]
    y_test = y[-n_test:]
    
    print(f"📊 Data Split: Train={len(X_train):,}, Test={len(X_test):,}")
    
    # Compute class weights for balanced training
    class_counts = np.bincount(y_train)
    class_weights = len(y_train) / (2 * class_counts)
    
    print(f"⚖️  Class Weights: Not Home Win={class_weights[0]:.2f}, Home Win={class_weights[1]:.2f}")
    
    # ── Base Estimators ──
    
    # 1. XGBoost
    xgb_params = {
        'n_estimators': 200,
        'max_depth': 5,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'random_state': 42,
        'n_jobs': -1
    }
    xgb_clf = XGBClassifier(**xgb_params)
    
    # 2. Random Forest
    rf_clf = RandomForestClassifier(
        n_estimators=200, 
        max_depth=10, 
        class_weight='balanced', 
        random_state=42, 
        n_jobs=-1
    )
    
    # 3. Logistic Regression
    lr_clf = Pipeline([
        ('scaler', RobustScaler()),
        ('imputer', SimpleImputer(strategy='median')),
        ('lr', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ])
    
    # Define the stack
    estimators = [
        ('xgb', xgb_clf),
        ('rf', rf_clf),
        ('lr', lr_clf)
    ]
    
    # Meta-learner
    meta_clf = LogisticRegression(class_weight='balanced', random_state=42)
    
    # Stacking Classifier
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_clf,
        cv=5,
        stack_method='predict_proba',
        n_jobs=-1
    )
    
    print("\n🚀 Training Binary Stacking Ensemble...")
    
    # Impute
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    
    stack.fit(X_train_imp, y_train)
    
    print("✓ Training complete!")
    
    # Evaluate
    y_pred = stack.predict(X_test_imp)
    y_prob = stack.predict_proba(X_test_imp)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    
    print("\n" + "-"*70)
    print(f"BINARY STACKING ENSEMBLE PERFORMANCE (Test Set)")
    print("-"*70)
    print(f"Overall Accuracy: {acc*100:.2f}%")
    
    outcome_names = ["Not Home Win", "Home Win"]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=outcome_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Extract Home Win Recall
    actual_home_wins = (y_test == 1).sum()
    correct_home_wins = ((y_test == 1) & (y_pred == 1)).sum()
    home_win_recall = correct_home_wins / actual_home_wins if actual_home_wins > 0 else 0
    print(f"\n🎯 Home Win Recall: {home_win_recall*100:.2f}%")
    
    return stack, acc, home_win_recall

def main():
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    
    X, y, cols = load_data(data_dir)
    if X is None:
        return
    
    stack_model, acc, home_win_recall = train_stacked_model(X, y, cols)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Binary Accuracy:      {acc*100:.2f}%")
    print(f"Home Win Recall:      {home_win_recall*100:.2f}%")
    
    if acc > 0.60:
        print("\n✅ SUCCESS: The Stacking Ensemble achieved good binary accuracy!")
    else:
        print("\n⚠️  WARNING: Accuracy is still below 60%. Consider further feature tuning.")

if __name__ == "__main__":
    main()
