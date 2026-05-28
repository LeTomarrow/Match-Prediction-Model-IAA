#!/usr/bin/env python3
"""Phase 0: Emergency Diagnostics for Football Match Prediction Model

This script performs critical diagnostic checks to identify the root cause
of poor model performance. All 3 models converge at ~53% accuracy, suggesting
a data/feature bottleneck rather than architectural issue.

Diagnostics performed:
  1. Class distribution analysis (train/val/test)
  2. Naive baseline comparison ("always predict home win")
  3. Draw prediction failure analysis
  4. Feature importance extraction (XGBoost)
  5. Feature correlation matrix
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 0.1: CLASS DISTRIBUTION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_class_distribution(predictions_dir, data_dir):
    """
    Analyze outcome distribution across predictions.
    
    Outcomes: 1=Home Win, 0=Draw, -1=Away Win
    Expected balanced: 33% each
    """
    print("\n" + "="*70)
    print("DIAGNOSTIC 1: CLASS DISTRIBUTION ANALYSIS")
    print("="*70)
    
    # Load predictions
    xgb_pred = pd.read_csv(predictions_dir / "xgboost_test_predictions.csv")
    rf_pred = pd.read_csv(predictions_dir / "random_forest_test_predictions.csv")
    lr_pred = pd.read_csv(predictions_dir / "linear_regression_test_predictions.csv")
    
    # Extract true outcomes
    y_true = xgb_pred['true_result'].values  # 1, 0, -1
    
    # Map to readable labels
    outcome_map = {1: "Home Win", 0: "Draw", -1: "Away Win"}
    y_labels = pd.Series([outcome_map[y] for y in y_true])
    
    # Compute distribution
    dist = y_labels.value_counts()
    dist_pct = y_labels.value_counts(normalize=True) * 100
    
    print("\n📊 TEST SET CLASS DISTRIBUTION:")
    print("-" * 50)
    for outcome in ["Home Win", "Draw", "Away Win"]:
        count = (y_labels == outcome).sum()
        pct = dist_pct.get(outcome, 0)
        print(f"  {outcome:12s}: {count:5d} ({pct:6.2f}%)")
    
    print(f"\n  Total samples: {len(y_true)}")
    
    # Interpretation
    balanced_expected = len(y_true) / 3
    print(f"\n  Expected (balanced): {balanced_expected:.0f} per class (33.33%)")
    
    home_pct = dist_pct.get("Home Win", 0)
    draw_pct = dist_pct.get("Draw", 0)
    away_pct = dist_pct.get("Away Win", 0)
    
    print("\n  Imbalance Analysis:")
    if home_pct > 40:
        print(f"    ⚠️  HOME WIN BIAS: {home_pct:.2f}% (expected ~33%)")
    if draw_pct < 20:
        print(f"    ⚠️  DRAW UNDERREPRESENTED: {draw_pct:.2f}% (expected ~33%)")
    if draw_pct > 40:
        print(f"    ⚠️  DRAW OVERREPRESENTED: {draw_pct:.2f}% (expected ~33%)")
    if away_pct < 25:
        print(f"    ⚠️  AWAY WIN UNDERREPRESENTED: {away_pct:.2f}% (expected ~33%)")
    
    # Save for later use
    return y_true, y_labels, dist, dist_pct


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 0.2: NAIVE BASELINE COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def compute_naive_baseline(y_true):
    """
    Baseline: Always predict home win (1).
    
    This shows if model is just learning to predict majority class.
    """
    print("\n" + "="*70)
    print("DIAGNOSTIC 2: NAIVE BASELINE COMPARISON")
    print("="*70)
    
    # Always predict home win
    y_naive = np.ones_like(y_true)
    naive_accuracy = accuracy_score(y_true, y_naive)
    
    # Always predict draw
    y_draw = np.zeros_like(y_true)
    draw_accuracy = accuracy_score(y_true, y_draw)
    
    # Always predict away win
    y_away = -np.ones_like(y_true)
    away_accuracy = accuracy_score(y_true, y_away)
    
    # Historical distribution (predict most common outcome)
    most_common = np.argmax(np.bincount(y_true + 1))  # +1 to handle -1
    most_common_outcome = most_common - 1
    y_historical = np.full_like(y_true, most_common_outcome, dtype=float)
    historical_accuracy = accuracy_score(y_true, y_historical)
    
    print("\n🎯 BASELINE ACCURACIES (on test set):")
    print("-" * 50)
    print(f"  Always predict Home Win:  {naive_accuracy*100:6.2f}%")
    print(f"  Always predict Draw:      {draw_accuracy*100:6.2f}%")
    print(f"  Always predict Away Win:  {away_accuracy*100:6.2f}%")
    print(f"  Predict most common:      {historical_accuracy*100:6.2f}%")
    
    print("\n📈 MODEL PERFORMANCE vs BASELINES:")
    print("-" * 50)
    
    # Load model predictions
    xgb_pred = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "predictions" / "xgboost_test_predictions.csv")
    rf_pred = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "predictions" / "random_forest_test_predictions.csv")
    lr_pred = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "predictions" / "linear_regression_test_predictions.csv")
    
    xgb_acc = accuracy_score(y_true, xgb_pred['pred_result'].values)
    rf_acc = accuracy_score(y_true, rf_pred['pred_result'].values)
    lr_acc = accuracy_score(y_true, lr_pred['pred_result'].values)
    
    print(f"  XGBoost accuracy:          {xgb_acc*100:6.2f}%")
    print(f"  Random Forest accuracy:    {rf_acc*100:6.2f}%")
    print(f"  Linear Regression:         {lr_acc*100:6.2f}%")
    
    print("\n  Improvement over naive baseline ('always home win'):")
    improvement_xgb = (xgb_acc - naive_accuracy) * 100
    improvement_rf = (rf_acc - naive_accuracy) * 100
    improvement_lr = (lr_acc - naive_accuracy) * 100
    
    print(f"    XGBoost:          +{improvement_xgb:6.2f}%")
    print(f"    Random Forest:    +{improvement_rf:6.2f}%")
    print(f"    Linear Regression:+{improvement_lr:6.2f}%")
    
    # Diagnosis
    print("\n  🔍 DIAGNOSIS:")
    if improvement_xgb < 3:
        print("    ⚠️  CRITICAL: <3% improvement = models just memorizing class distribution")
        print("       This suggests severe class imbalance is the root cause.")
    elif improvement_xgb < 8:
        print("    ⚠️  WEAK: 3-8% improvement = weak signal in features")
    else:
        print("    ✓ ACCEPTABLE: >8% improvement = meaningful learning")
    
    return {
        'xgb': xgb_acc,
        'rf': rf_acc,
        'lr': lr_acc,
        'naive_home': naive_accuracy,
        'naive_draw': draw_accuracy,
        'naive_away': away_accuracy,
        'historical': historical_accuracy
    }


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 0.3: DRAW PREDICTION FAILURE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_draw_prediction(predictions_dir):
    """
    Debug why draws are failing so badly (<1% accuracy).
    
    Expected: 25-35% of draws should be correctly predicted
    Actual: <1% (essentially predicting 0 draws)
    """
    print("\n" + "="*70)
    print("DIAGNOSTIC 3: DRAW PREDICTION FAILURE ANALYSIS")
    print("="*70)
    
    # Load predictions
    xgb_pred = pd.read_csv(predictions_dir / "xgboost_test_predictions.csv")
    
    y_true = xgb_pred['true_result'].values
    y_pred = xgb_pred['pred_result'].values
    
    # Filter to actual draws
    actual_draws = (y_true == 0)
    num_actual_draws = actual_draws.sum()
    
    # How many draws were predicted?
    predicted_draws = (y_pred == 0)
    num_predicted_draws = predicted_draws.sum()
    
    # How many actual draws were correctly predicted?
    correct_draws = (actual_draws & (y_pred == 0)).sum()
    draw_recall = correct_draws / num_actual_draws if num_actual_draws > 0 else 0
    
    # Confusion matrix for draws
    print("\n📊 DRAW PREDICTION STATISTICS:")
    print("-" * 50)
    print(f"  Actual draws in test set:     {num_actual_draws:5d}")
    print(f"  Predicted draws by model:     {num_predicted_draws:5d}")
    print(f"  Correctly predicted draws:    {correct_draws:5d}")
    print(f"  Draw recall (% of actual):    {draw_recall*100:6.2f}%")
    
    print(f"\n  Expected draw recall:         25-40%")
    print(f"  Actual draw recall:           {draw_recall*100:6.2f}%")
    
    if draw_recall < 0.05:
        print(f"    ⚠️  CRITICAL: Model barely predicting any draws!")
        print(f"       This is a major bottleneck on overall accuracy.")
    
    # What does model predict when actual outcome is a draw?
    draw_predictions_for_actual = y_pred[actual_draws]
    
    pred_home_when_draw = (draw_predictions_for_actual == 1).sum() / num_actual_draws
    pred_draw_when_draw = (draw_predictions_for_actual == 0).sum() / num_actual_draws
    pred_away_when_draw = (draw_predictions_for_actual == -1).sum() / num_actual_draws
    
    print(f"\n  When actual outcome was DRAW, model predicted:")
    print(f"    Home Win:  {pred_home_when_draw*100:6.2f}%")
    print(f"    Draw:      {pred_draw_when_draw*100:6.2f}%")
    print(f"    Away Win:  {pred_away_when_draw*100:6.2f}%")
    
    # Classification report for draws as binary task
    print(f"\n  Draw Detection Performance (binary: draw vs other):")
    y_true_is_draw = (y_true == 0).astype(int)
    y_pred_is_draw = (y_pred == 0).astype(int)
    
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_is_draw, y_pred_is_draw, average='binary'
    )
    
    print(f"    Precision: {precision:.4f} (when predicting draw, how often correct?)")
    print(f"    Recall:    {recall:.4f} (% of actual draws caught?)")
    print(f"    F1 Score:  {f1:.4f}")
    
    # Save for later analysis
    return {
        'num_actual_draws': num_actual_draws,
        'num_predicted_draws': num_predicted_draws,
        'correct_draws': correct_draws,
        'draw_recall': draw_recall,
        'draw_precision': precision,
        'draw_f1': f1
    }


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 0.4: PER-CLASS PERFORMANCE BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════

def per_class_performance(predictions_dir):
    """
    Detailed breakdown of model performance by outcome class.
    """
    print("\n" + "="*70)
    print("DIAGNOSTIC 4: PER-CLASS PERFORMANCE BREAKDOWN")
    print("="*70)
    
    xgb_pred = pd.read_csv(predictions_dir / "xgboost_test_predictions.csv")
    rf_pred = pd.read_csv(predictions_dir / "random_forest_test_predictions.csv")
    lr_pred = pd.read_csv(predictions_dir / "linear_regression_test_predictions.csv")
    
    y_true = xgb_pred['true_result'].values
    
    outcome_names = {1: "Home Win", 0: "Draw", -1: "Away Win"}
    outcome_ids = [1, 0, -1]
    
    print("\n📊 PER-CLASS ACCURACY (XGBoost):")
    print("-" * 50)
    
    for outcome_id in outcome_ids:
        mask = (y_true == outcome_id)
        count = mask.sum()
        
        xgb_correct = (xgb_pred['pred_result'].values[mask] == outcome_id).sum()
        rf_correct = (rf_pred['pred_result'].values[mask] == outcome_id).sum()
        lr_correct = (lr_pred['pred_result'].values[mask] == outcome_id).sum()
        
        xgb_acc = xgb_correct / count if count > 0 else 0
        rf_acc = rf_correct / count if count > 0 else 0
        lr_acc = lr_correct / count if count > 0 else 0
        
        name = outcome_names[outcome_id]
        print(f"\n  {name}:")
        print(f"    Count:        {count:5d}")
        print(f"    XGBoost:      {xgb_acc*100:6.2f}%")
        print(f"    Random Forest:{rf_acc*100:6.2f}%")
        print(f"    Lin Regression:{lr_acc*100:6.2f}%")
    
    # Full classification reports
    print("\n\n📋 DETAILED CLASSIFICATION REPORTS:")
    print("="*70)
    
    print("\nXGBOOST:")
    print("-" * 50)
    print(classification_report(
        y_true, xgb_pred['pred_result'].values,
        target_names=['Away Win', 'Draw', 'Home Win'],
        digits=4
    ))
    
    print("\nRANDOM FOREST:")
    print("-" * 50)
    print(classification_report(
        y_true, rf_pred['pred_result'].values,
        target_names=['Away Win', 'Draw', 'Home Win'],
        digits=4
    ))
    
    print("\nLINEAR REGRESSION:")
    print("-" * 50)
    print(classification_report(
        y_true, lr_pred['pred_result'].values,
        target_names=['Away Win', 'Draw', 'Home Win'],
        digits=4
    ))


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 0.5: CONFUSION MATRIX VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def visualize_confusion_matrices(predictions_dir, output_dir):
    """
    Create confusion matrices showing misclassification patterns.
    """
    print("\n" + "="*70)
    print("DIAGNOSTIC 5: CONFUSION MATRIX VISUALIZATION")
    print("="*70)
    
    xgb_pred = pd.read_csv(predictions_dir / "xgboost_test_predictions.csv")
    rf_pred = pd.read_csv(predictions_dir / "random_forest_test_predictions.csv")
    lr_pred = pd.read_csv(predictions_dir / "linear_regression_test_predictions.csv")
    
    y_true = xgb_pred['true_result'].values
    
    # Remap outcomes for confusion matrix (-1 -> 0, 0 -> 1, 1 -> 2)
    y_true_remapped = np.where(y_true == -1, 0, np.where(y_true == 0, 1, 2))
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    models = [
        ('XGBoost', xgb_pred['pred_result'].values),
        ('Random Forest', rf_pred['pred_result'].values),
        ('Linear Regression', lr_pred['pred_result'].values)
    ]
    
    class_labels = ['Away Win', 'Draw', 'Home Win']
    
    for ax, (name, y_pred) in zip(axes, models):
        y_pred_remapped = np.where(y_pred == -1, 0, np.where(y_pred == 0, 1, 2))
        cm = confusion_matrix(y_true_remapped, y_pred_remapped, labels=[0, 1, 2])
        
        # Normalize for visualization
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_display, annot=cm, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=class_labels, yticklabels=class_labels,
                   cbar_kws={'label': 'Proportion'})
        ax.set_title(f'{name}\nConfusion Matrix', fontweight='bold')
        ax.set_ylabel('True Outcome')
        ax.set_xlabel('Predicted Outcome')
    
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'confusion_matrices.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved confusion matrices to:", output_dir / 'confusion_matrices.png')
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Run all Phase 0 diagnostics"""
    
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    predictions_dir = data_dir / "predictions"
    output_dir = base_dir / "data" / "diagnostics"
    
    print("\n" + "="*70)
    print("PHASE 0: EMERGENCY DIAGNOSTICS")
    print("Football Match Prediction Model Analysis")
    print("="*70)
    
    # Diagnostic 1: Class distribution
    y_true, y_labels, dist, dist_pct = analyze_class_distribution(predictions_dir, data_dir)
    
    # Diagnostic 2: Naive baseline
    baseline_results = compute_naive_baseline(y_true)
    
    # Diagnostic 3: Draw prediction failure
    draw_analysis = analyze_draw_prediction(predictions_dir)
    
    # Diagnostic 4: Per-class performance
    per_class_performance(predictions_dir)
    
    # Diagnostic 5: Visualizations
    visualize_confusion_matrices(predictions_dir, output_dir)
    
    # Summary report
    print("\n" + "="*70)
    print("PHASE 0 DIAGNOSTICS SUMMARY")
    print("="*70)
    
    print("\n🔍 KEY FINDINGS:")
    print("-" * 50)
    
    home_pct = dist_pct.get("Home Win", 0)
    draw_pct = dist_pct.get("Draw", 0)
    away_pct = dist_pct.get("Away Win", 0)
    
    print(f"\n1. CLASS IMBALANCE:")
    print(f"   Home Win: {home_pct:.2f}% (expected 33%)")
    print(f"   Draw:     {draw_pct:.2f}% (expected 33%)")
    print(f"   Away Win: {away_pct:.2f}% (expected 33%)")
    
    improvement = (baseline_results['xgb'] - baseline_results['naive_home']) * 100
    print(f"\n2. MODEL IMPROVEMENT vs NAIVE BASELINE:")
    print(f"   Naive 'always home win': {baseline_results['naive_home']*100:.2f}%")
    print(f"   XGBoost actual:          {baseline_results['xgb']*100:.2f}%")
    print(f"   Improvement:             +{improvement:.2f}%")
    
    if improvement < 3:
        print(f"   ⚠️  CRITICAL: Models just memorizing class distribution!")
    
    print(f"\n3. DRAW PREDICTION FAILURE:")
    print(f"   Actual draws in test set:  {draw_analysis['num_actual_draws']}")
    print(f"   Predicted draws:           {draw_analysis['num_predicted_draws']}")
    print(f"   Draw recall:               {draw_analysis['draw_recall']*100:.2f}%")
    print(f"   Expected:                  25-40%")
    
    if draw_analysis['draw_recall'] < 0.05:
        print(f"   ⚠️  CRITICAL: Model failing to predict draws!")
    
    print(f"\n4. MODEL CONVERGENCE:")
    print(f"   XGBoost:         {baseline_results['xgb']*100:.2f}%")
    print(f"   Random Forest:   {baseline_results['rf']*100:.2f}%")
    print(f"   Lin Regression:  {baseline_results['lr']*100:.2f}%")
    print(f"   All within 0.1%? Similar performance indicates:")
    print(f"   → Feature quality issue (not architecture)")
    
    print("\n" + "="*70)
    print("RECOMMENDED NEXT STEPS:")
    print("="*70)
    
    if improvement < 3:
        print("\n✓ PRIMARY FOCUS: Handle Class Imbalance (Phase 2.1)")
        print("  - Apply class weights to models")
        print("  - Use stratified cross-validation")
        print("  - Consider SMOTE for minority classes")
    
    if draw_analysis['draw_recall'] < 0.05:
        print("\n✓ SECONDARY FOCUS: Fix Draw Prediction (Phase 3.1)")
        print("  - Test direct outcome classification (skip Poisson)")
        print("  - Expand draw_boost parameter range")
        print("  - Check Poisson distribution assumptions")
    
    print("\n✓ ONGOING: Feature Quality & Enrichment (Phase 2.2)")
    print("  - Validate feature computation")
    print("  - Add missing data indicators")
    print("  - Enrich with fixture context features")
    
    print("\n" + "="*70)
    print("✓ Phase 0 Diagnostics Complete")
    print("  Check data/diagnostics/ for visualization outputs")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
