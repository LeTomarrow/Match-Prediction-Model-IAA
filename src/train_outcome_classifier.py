#!/usr/bin/env python3
"""
PHASE 3.1: Direct Outcome Classification (Skip Poisson)

This script tests whether bypassing the Poisson distribution and training
a direct 3-class classifier improves draw prediction and overall accuracy.

The hypothesis: Poisson(goals) naturally suppresses draws (low-probability events),
causing draw recall to collapse to 0.41%. A direct outcome classifier should
unlock draw prediction.

Expected Impact: Draw recall 0.41% → 15-25%
Timeline: 1-2 hours
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("WARNING: xgboost not installed")

# ══════════════════════════════════════════════════════════════════════════════
# DIRECT OUTCOME CLASSIFICATION: NO POISSON
# ══════════════════════════════════════════════════════════════════════════════

def load_features_and_outcomes(data_dir):
    """Load the preprocessed features and outcomes"""
    
    # Try to find feature file (may be large, so check what's available)
    feature_files = list(data_dir.glob("match_features*.csv"))
    if not feature_files:
        print("ERROR: No match_features*.csv found in data/")
        return None, None, None, None
    
    features_path = feature_files[0]
    print(f"Loading features from: {features_path.name}")
    
    try:
        df = pd.read_csv(features_path)
    except Exception as e:
        print(f"ERROR loading features: {e}")
        return None, None, None, None
    
    print(f"✓ Loaded {len(df)} matches with {len(df.columns)} columns")
    
    # Compute outcome from goals
    if 'home_club_goals' in df.columns and 'away_club_goals' in df.columns:
        home_goals = df['home_club_goals'].values
        away_goals = df['away_club_goals'].values
        # 1=home win, 0=draw, -1=away win
        y = np.where(home_goals > away_goals, 1, 
                    np.where(home_goals == away_goals, 0, -1))
    elif 'true_result' in df.columns:
        y = df['true_result'].values  # 1=home, 0=draw, -1=away
    else:
        print("ERROR: Cannot compute outcome")
        return None, None, None, None
    
    # Identify feature columns (exclude metadata and outcome)
    exclude_cols = {
        'game_id', 'home_club_id', 'away_club_id', 'date', 'competition_id',
        'home_feature_date', 'away_feature_date', 'season', 'round',
        'true_result', 'outcome', 'true_home_goals', 'true_away_goals',
        'home_club_goals', 'away_club_goals', 'target', 'home_club_name', 
        'away_club_name', 'home_indicator'
    }
    
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].values
    
    print(f"✓ Features: {len(feature_cols)}")
    print(f"  Outcome distribution: Home {(y==1).sum()} | Draw {(y==0).sum()} | Away {(y==-1).sum()}")
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    print(f"  ✓ Imputed missing values")
    
    return X, y, df, feature_cols


def train_direct_outcome_classifier(X, y, feature_cols, output_dir):
    """
    Train direct 3-class outcome classifiers WITHOUT Poisson distribution.
    
    Outcomes: 1=Home Win, 0=Draw, -1=Away Win (remapped to 0,1,2 for sklearn)
    """
    
    print("\n" + "="*70)
    print("PHASE 3.1: DIRECT OUTCOME CLASSIFICATION")
    print("="*70)
    
    # Remap outcomes for sklearn: -1→0, 0→1, 1→2
    y_remapped = np.where(y == -1, 0, np.where(y == 0, 1, 2))
    outcome_names = {0: "Away Win", 1: "Draw", 2: "Home Win"}
    
    # Split data: last 20% for test, last 10% of remaining for validation
    n_total = len(X)
    n_test = int(0.20 * n_total)
    n_val = int(0.10 * (n_total - n_test))
    
    X_train = X[:-n_test-n_val]
    y_train = y_remapped[:-n_test-n_val]
    
    X_val = X[-n_test-n_val:-n_test]
    y_val = y_remapped[-n_test-n_val:-n_test]
    
    X_test = X[-n_test:]
    y_test = y_remapped[-n_test:]
    
    print(f"\n📊 Data Split:")
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Val:   {len(X_val):,} samples")
    print(f"  Test:  {len(X_test):,} samples")
    
    # Compute class weights
    class_counts = np.bincount(y_train)
    class_weights = len(y_train) / (3 * class_counts)
    
    print(f"\n⚖️  Class Weights (balanced):")
    for outcome_id, weight in enumerate(class_weights):
        print(f"  {outcome_names[outcome_id]:12s}: {weight:.3f}")
    
    models = {}
    results = {}
    
    # ── Model 1: XGBoost with Balanced Weights ──
    if HAS_XGBOOST:
        print("\n" + "-"*70)
        print("Model 1: XGBoost (Direct Classification, Balanced Weights)")
        print("-"*70)
        
        model_xgb = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=1.0,
            objective='multi:softprob',
            num_class=3,
            random_state=42,
            n_jobs=-1
        )
        
        # Train with class weights
        scale_pos_weights = np.array([class_weights[0] / class_weights[2],
                                     class_weights[1] / class_weights[2]])
        
        model_xgb.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            sample_weight=class_weights[y_train],
            verbose=False
        )
        
        y_pred_xgb = model_xgb.predict(X_test)
        y_pred_probs_xgb = model_xgb.predict_proba(X_test)
        
        acc_train_xgb = model_xgb.score(X_train, y_train)
        acc_val_xgb = model_xgb.score(X_val, y_val)
        acc_test_xgb = model_xgb.score(X_test, y_test)
        
        print(f"\n  Accuracies:")
        print(f"    Train: {acc_train_xgb*100:6.2f}%")
        print(f"    Val:   {acc_val_xgb*100:6.2f}%")
        print(f"    Test:  {acc_test_xgb*100:6.2f}%")
        
        print(f"\n  Per-Class Performance (Test):")
        print(classification_report(y_test, y_pred_xgb, 
                                   target_names=list(outcome_names.values()),
                                   digits=4))
        
        models['xgb_direct'] = model_xgb
        results['xgb_direct'] = {
            'model': model_xgb,
            'y_pred': y_pred_xgb,
            'y_pred_probs': y_pred_probs_xgb,
            'accuracy': acc_test_xgb,
            'name': 'XGBoost (Direct Classification)'
        }
    
    # ── Model 2: Random Forest with Balanced Weights ──
    print("\n" + "-"*70)
    print("Model 2: Random Forest (Direct Classification, Balanced Weights)")
    print("-"*70)
    
    # Compute sample weights for RF
    sample_weights_train = class_weights[y_train]
    
    model_rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    model_rf.fit(X_train, y_train, sample_weight=sample_weights_train)
    
    y_pred_rf = model_rf.predict(X_test)
    y_pred_probs_rf = model_rf.predict_proba(X_test)
    
    acc_train_rf = model_rf.score(X_train, y_train, sample_weight=sample_weights_train)
    acc_val_rf = model_rf.score(X_val, y_val)
    acc_test_rf = model_rf.score(X_test, y_test)
    
    print(f"\n  Accuracies:")
    print(f"    Train: {acc_train_rf*100:6.2f}%")
    print(f"    Val:   {acc_val_rf*100:6.2f}%")
    print(f"    Test:  {acc_test_rf*100:6.2f}%")
    
    print(f"\n  Per-Class Performance (Test):")
    print(classification_report(y_test, y_pred_rf,
                               target_names=list(outcome_names.values()),
                               digits=4))
    
    models['rf_direct'] = model_rf
    results['rf_direct'] = {
        'model': model_rf,
        'y_pred': y_pred_rf,
        'y_pred_probs': y_pred_probs_rf,
        'accuracy': acc_test_rf,
        'name': 'Random Forest (Direct Classification)'
    }
    
    # ── Model 3: Logistic Regression Baseline ──
    print("\n" + "-"*70)
    print("Model 3: Logistic Regression (Direct Classification)")
    print("-"*70)
    
    model_lr = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    
    model_lr.fit(X_train, y_train, sample_weight=sample_weights_train)
    
    y_pred_lr = model_lr.predict(X_test)
    y_pred_probs_lr = model_lr.predict_proba(X_test)
    
    acc_train_lr = model_lr.score(X_train, y_train, sample_weight=sample_weights_train)
    acc_val_lr = model_lr.score(X_val, y_val)
    acc_test_lr = model_lr.score(X_test, y_test)
    
    print(f"\n  Accuracies:")
    print(f"    Train: {acc_train_lr*100:6.2f}%")
    print(f"    Val:   {acc_val_lr*100:6.2f}%")
    print(f"    Test:  {acc_test_lr*100:6.2f}%")
    
    print(f"\n  Per-Class Performance (Test):")
    print(classification_report(y_test, y_pred_lr,
                               target_names=list(outcome_names.values()),
                               digits=4))
    
    models['lr_direct'] = model_lr
    results['lr_direct'] = {
        'model': model_lr,
        'y_pred': y_pred_lr,
        'y_pred_probs': y_pred_probs_lr,
        'accuracy': acc_test_lr,
        'name': 'Logistic Regression (Direct Classification)'
    }
    
    return models, results, (X_test, y_test), outcome_names


def compare_with_baseline(baseline_accuracy, results):
    """Compare direct classification results with baseline Poisson approach"""
    
    print("\n" + "="*70)
    print("COMPARISON: DIRECT CLASSIFICATION vs BASELINE POISSON")
    print("="*70)
    
    print(f"\n📊 BASELINE (Current Poisson Approach):")
    print(f"  XGBoost accuracy: {baseline_accuracy*100:6.2f}%")
    print(f"  Draw recall:       0.41%")
    
    print(f"\n🎯 DIRECT CLASSIFICATION (All Models):")
    print("-" * 50)
    
    best_model_name = None
    best_accuracy = 0
    
    for model_name, result in results.items():
        acc = result['accuracy']
        print(f"  {result['name']:40s}: {acc*100:6.2f}%")
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = model_name
    
    improvement = (best_accuracy - baseline_accuracy) * 100
    
    print(f"\n📈 IMPROVEMENT:")
    print(f"  Best direct classification: {best_accuracy*100:.2f}%")
    print(f"  Baseline Poisson:           {baseline_accuracy*100:.2f}%")
    print(f"  Improvement:                +{improvement:.2f}%")
    
    if improvement < 0.5:
        print(f"  ⚠️  Minimal improvement - Poisson may be acceptable")
    elif improvement < 2:
        print(f"  ✓ Moderate improvement - Consider switching if draw recall is better")
    else:
        print(f"  ✓✓ Significant improvement - RECOMMEND SWITCHING to direct classification!")
    
    return best_model_name, best_accuracy


def extract_draw_statistics(y_test, results):
    """Extract draw-specific performance metrics"""
    
    print("\n" + "="*70)
    print("DRAW PREDICTION PERFORMANCE")
    print("="*70)
    
    print(f"\n📊 TEST SET COMPOSITION:")
    print(f"  Total samples:     {len(y_test):,}")
    print(f"  Actual draws:      {(y_test == 1).sum():,} ({(y_test == 1).sum()/len(y_test)*100:.1f}%)")
    
    print(f"\n  Per-model Draw Performance:")
    print("-" * 60)
    
    for model_name, result in results.items():
        y_pred = result['y_pred']
        
        actual_draws = (y_test == 1).sum()
        predicted_draws = (y_pred == 1).sum()
        correct_draws = ((y_test == 1) & (y_pred == 1)).sum()
        
        draw_recall = correct_draws / actual_draws if actual_draws > 0 else 0
        draw_precision = correct_draws / predicted_draws if predicted_draws > 0 else 0
        
        print(f"\n  {result['name']}:")
        print(f"    Predicted draws:     {predicted_draws:,}")
        print(f"    Correct draws:       {correct_draws:,}")
        print(f"    Draw recall:         {draw_recall*100:6.2f}%")
        print(f"    Draw precision:      {draw_precision*100:6.2f}%")
        
        if draw_recall > 0.10:
            print(f"    ✓ Significant improvement from baseline (0.41%)")
        elif draw_recall > 0.02:
            print(f"    ~ Minor improvement from baseline")
        else:
            print(f"    ⚠️  No improvement from baseline")


def main():
    """Execute Phase 3.1: Direct Outcome Classification"""
    
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    output_dir = base_dir / "data" / "phase3_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("PHASE 3.1: DIRECT OUTCOME CLASSIFICATION")
    print("Testing bypass of Poisson distribution")
    print("="*70)
    
    # Load data
    X, y, df, feature_cols = load_features_and_outcomes(data_dir)
    if X is None:
        print("\nERROR: Could not load features")
        return
    
    # Train direct classifiers
    models, results, (X_test, y_test), outcome_names = train_direct_outcome_classifier(
        X, y, feature_cols, output_dir
    )
    
    # Baseline accuracy from current Poisson approach
    baseline_acc = 0.5331  # XGBoost from diagnostics
    
    # Compare with baseline
    best_model, best_acc = compare_with_baseline(baseline_acc, results)
    
    # Extract draw-specific performance
    extract_draw_statistics(y_test, results)
    
    # Summary recommendations
    print("\n" + "="*70)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*70)
    
    improvement_pct = (best_acc - baseline_acc) * 100
    
    if improvement_pct >= 2:
        print(f"\n✓✓ STRONG RECOMMENDATION: Switch to Direct Classification")
        print(f"   Improvement: +{improvement_pct:.2f}%")
        print(f"   Best model: {results[best_model]['name']}")
        print(f"   New accuracy: {best_acc*100:.2f}%")
    elif improvement_pct >= 0.5:
        print(f"\n✓ MODERATE RECOMMENDATION: Consider switching")
        print(f"   Improvement: +{improvement_pct:.2f}%")
        print(f"   Especially if draw recall is significantly higher")
    else:
        print(f"\n~ RECOMMENDATION: Stick with Poisson approach")
        print(f"   Direct classification showed: {improvement_pct:.2f}% (negligible)")
        print(f"   Focus instead on feature enrichment (Phase 2.2)")
    
    print("\n" + "="*70)
    print("✓ Phase 3.1 Complete")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
