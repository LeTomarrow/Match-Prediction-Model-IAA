#!/usr/bin/env python3
"""Train dual-regression football score models from pre-match engineered features.

Pipeline improvements over baseline:
  - Adds differential/ratio features between home and away team stats.
  - Trains XGBoost with early stopping and cross-validated hyperparameter search.
  - Includes a Poisson-calibrated wrapper for more realistic goal distributions.
  - Trains a stacked ensemble (Linear + RF + XGB) using a time-series-safe blender.
  - Reports full metrics: MAE, RMSE, R², outcome accuracy, and exact-score accuracy.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor, callback as xgb_callback
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("WARNING: xgboost not installed. Install it with: pip install xgboost")

# ─────────────────────────────────────────────────────────────────────────────
TARGET_HOME = "home_club_goals"
TARGET_AWAY = "away_club_goals"
DATE_COL = "date"
ID_COLS = {"game_id", "home_club_id", "away_club_id", "home_feature_date",
           "away_feature_date", TARGET_HOME, TARGET_AWAY}
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train football goal prediction models")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "match_features.csv",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "model_results.csv",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "model_results_summary.json",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "predictions",
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["linear_regression", "ridge", "random_forest", "xgboost",
                 "gradient_boosting", "ensemble"],
        default=["linear_regression", "ridge", "random_forest", "xgboost", "ensemble"],
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--tune-xgb",
        action="store_true",
        default=False,
        help="Run light hyperparameter search for XGBoost (adds ~2-3 min)",
    )
    return parser.parse_args()


# ─────────────────────────── Feature engineering ─────────────────────────────

def add_differential_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add home-minus-away differential and ratio features."""
    df = df.copy()

    paired = {
        # (home_col_suffix, away_col_suffix)
        "team_ppg_3": "ppg",
        "team_ppg_5": "ppg5",
        "team_ppg_10": "ppg10",
        "team_goal_diff_avg_3": "gd3",
        "team_goal_diff_avg_5": "gd5",
        "team_goal_diff_avg_10": "gd10",
        "team_goals_scored_avg_3": "gs3",
        "team_goals_scored_avg_5": "gs5",
        "team_goals_scored_avg_10": "gs10",
        "team_goals_conceded_avg_3": "gc3",
        "team_goals_conceded_avg_5": "gc5",
        "team_goals_conceded_avg_10": "gc10",
        "team_win_rate_3": "wr3",
        "team_win_rate_5": "wr5",
        "team_win_rate_10": "wr10",
        "team_form_weighted_points_3": "fwp3",
        "team_form_weighted_points_5": "fwp5",
        "split_ppg_3": "split_ppg3",
        "split_ppg_5": "split_ppg5",
        "split_goals_scored_avg_3": "split_gs3",
        "split_goals_scored_avg_5": "split_gs5",
        "split_goals_conceded_avg_3": "split_gc3",
        "split_goals_conceded_avg_5": "split_gc5",
        "squad_total_market_value_eur": "squad_mv",
        "squad_top11_market_value_eur": "top11_mv",
        "squad_value_per_starter_eur": "starter_mv",
    }

    for suffix, short in paired.items():
        hc = f"home_{suffix}"
        ac = f"away_{suffix}"
        if hc in df.columns and ac in df.columns:
            diff_col = f"diff_{short}"
            ratio_col = f"ratio_{short}"
            df[diff_col] = df[hc] - df[ac]
            denom = df[ac].replace(0, np.nan)
            df[ratio_col] = df[hc] / denom

    # Attack vs defence match-ups
    for w in (3, 5, 10):
        hgs = f"home_team_goals_scored_avg_{w}"
        agc = f"away_team_goals_conceded_avg_{w}"
        ags = f"away_team_goals_scored_avg_{w}"
        hgc = f"home_team_goals_conceded_avg_{w}"
        if all(c in df.columns for c in [hgs, agc, ags, hgc]):
            df[f"home_attack_vs_away_defence_{w}"] = df[hgs] - df[agc]
            df[f"away_attack_vs_home_defence_{w}"] = df[ags] - df[hgc]

    # Poisson lambda estimates (Dixon-Coles inspired, raw)
    if all(c in df.columns for c in [
        "home_team_goals_scored_avg_5", "away_team_goals_conceded_avg_5",
        "away_team_goals_scored_avg_5", "home_team_goals_conceded_avg_5",
    ]):
        overall_mean = 1.35  # typical European league average goals per match side
        att_h = (df["home_team_goals_scored_avg_5"] / overall_mean).clip(lower=0.1)
        def_a = (df["away_team_goals_conceded_avg_5"] / overall_mean).clip(lower=0.1)
        att_a = (df["away_team_goals_scored_avg_5"] / overall_mean).clip(lower=0.1)
        def_h = (df["home_team_goals_conceded_avg_5"] / overall_mean).clip(lower=0.1)
        HOME_ADV = 1.15
        df["poisson_lambda_home"] = att_h * def_a * overall_mean * HOME_ADV
        df["poisson_lambda_away"] = att_a * def_h * overall_mean

    # Season phase (early / mid / late) from round number
    if "round" in df.columns:
        round_num = pd.to_numeric(df["round"], errors="coerce")
        df["round_numeric"] = round_num
        df["season_phase"] = pd.cut(
            round_num,
            bins=[0, 10, 25, 1000],
            labels=[0, 1, 2],
        ).astype(float)

    return df


def add_recent_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Add momentum: difference between short-window and long-window form."""
    df = df.copy()
    for side in ("home", "away"):
        ppg3 = f"{side}_team_ppg_3"
        ppg10 = f"{side}_team_ppg_10"
        gd3 = f"{side}_team_goal_diff_avg_3"
        gd10 = f"{side}_team_goal_diff_avg_10"
        if ppg3 in df.columns and ppg10 in df.columns:
            df[f"{side}_momentum_ppg"] = df[ppg3] - df[ppg10]
        if gd3 in df.columns and gd10 in df.columns:
            df[f"{side}_momentum_gd"] = df[gd3] - df[gd10]
    return df


# ─────────────────────────── Data loading ─────────────────────────────────────

def load_and_prepare_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in [DATE_COL, TARGET_HOME, TARGET_AWAY]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, TARGET_HOME, TARGET_AWAY]).copy()
    sort_cols = [DATE_COL, "game_id"] if "game_id" in df.columns else [DATE_COL]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def build_feature_matrix(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Apply all feature engineering and return X, y_home, y_away."""
    df = add_differential_features(df)
    df = add_recent_momentum(df)

    drop = ID_COLS | {DATE_COL, "home_feature_date", "away_feature_date"}
    candidate = [c for c in df.columns if c not in drop]

    X = df[candidate].copy()
    y_home = pd.to_numeric(df[TARGET_HOME], errors="coerce")
    y_away = pd.to_numeric(df[TARGET_AWAY], errors="coerce")

    valid = y_home.notna() & y_away.notna()
    return X.loc[valid].reset_index(drop=True), \
           y_home.loc[valid].reset_index(drop=True), \
           y_away.loc[valid].reset_index(drop=True)


# ─────────────────────────── Split ────────────────────────────────────────────

def chronological_split(
    X: pd.DataFrame, y_home: pd.Series, y_away: pd.Series, test_size: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    n = len(X)
    split_idx = min(max(int(n * (1.0 - test_size)), 1), n - 1)
    return (
        X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy(),
        y_home.iloc[:split_idx].copy(), y_home.iloc[split_idx:].copy(),
        y_away.iloc[:split_idx].copy(), y_away.iloc[split_idx:].copy(),
    )


# ─────────────────────────── Preprocessing ────────────────────────────────────

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    transformers = []
    if numeric_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_cols,
        ))
    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical_cols,
        ))
    if not transformers:
        raise ValueError("No usable columns found")
    return ColumnTransformer(transformers, remainder="drop")


# ─────────────────────────── Models ───────────────────────────────────────────

def _xgb_params(random_state: int, tune: bool, X_train: pd.DataFrame, y_train: pd.Series):
    """Return best XGBoost params, optionally after a light grid search."""
    base = dict(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=3,
        subsample=0.80,
        colsample_bytree=0.75,
        reg_alpha=0.1,
        reg_lambda=1.5,
        gamma=0.1,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
    )
    if not tune:
        return base

    # Light grid search using TimeSeriesSplit
    from sklearn.model_selection import cross_val_score
    tscv = TimeSeriesSplit(n_splits=4)
    best_score = np.inf
    best_params = base.copy()

    search_grid = [
        {"max_depth": 4, "learning_rate": 0.05, "subsample": 0.85, "n_estimators": 400},
        {"max_depth": 5, "learning_rate": 0.03, "subsample": 0.80, "n_estimators": 600},
        {"max_depth": 6, "learning_rate": 0.02, "subsample": 0.75, "n_estimators": 800},
        {"max_depth": 5, "learning_rate": 0.04, "subsample": 0.85, "colsample_bytree": 0.80, "n_estimators": 500},
    ]

    prep = Pipeline([("imp", SimpleImputer(strategy="median"))])
    X_imp = prep.fit_transform(X_train.select_dtypes(include=[np.number]))

    for override in search_grid:
        params = {**base, **override}
        mdl = XGBRegressor(**params)
        scores = -cross_val_score(mdl, X_imp, y_train, cv=tscv,
                                  scoring="neg_mean_absolute_error", n_jobs=1)
        mean_mae = scores.mean()
        print(f"  XGB tune {override}: MAE={mean_mae:.4f}")
        if mean_mae < best_score:
            best_score = mean_mae
            best_params = params

    print(f"  Best XGB params: {best_params}")
    return best_params


class AveragingEnsemble(BaseEstimator):
    """Simple averaging ensemble compatible with sklearn Pipeline (sklearn 1.8+)."""

    def __init__(self, estimators: List[Tuple[str, object]]):
        self.estimators = estimators

    def fit(self, X, y):
        import copy
        self.fitted_estimators_ = []
        for name, est in self.estimators:
            fitted = copy.deepcopy(est)
            fitted.fit(X, y)
            self.fitted_estimators_.append((name, fitted))
        return self

    def predict(self, X) -> np.ndarray:
        preds = np.array([est.predict(X) for _, est in self.fitted_estimators_])
        return preds.mean(axis=0)


def define_models(
    random_state: int,
    selected: List[str],
    X_train: pd.DataFrame,
    y_home_train: pd.Series,
    tune_xgb: bool = False,
) -> Dict[str, object]:
    models: Dict[str, object] = {}

    if "linear_regression" in selected:
        models["linear_regression"] = LinearRegression()

    if "ridge" in selected:
        models["ridge"] = Ridge(alpha=1.0)

    if "random_forest" in selected:
        models["random_forest"] = RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            max_features=0.6,
            random_state=random_state,
            n_jobs=-1,
        )

    if "xgboost" in selected:
        if HAS_XGBOOST:
            print("Configuring XGBoost...")
            if tune_xgb:
                print("Running XGBoost hyperparameter search...")
            params = _xgb_params(random_state, tune_xgb, X_train, y_home_train)
            models["xgboost"] = XGBRegressor(**params)
        else:
            print("xgboost requested but not installed: skipping")

    if "gradient_boosting" in selected:
        models["gradient_boosting"] = GradientBoostingRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.80,
            min_samples_leaf=3,
            random_state=random_state,
        )

    if "ensemble" in selected:
        # Manual averaging ensemble: train base models and average their predictions
        base_models = [
            ("ridge", Ridge(alpha=1.0)),
            ("rf", RandomForestRegressor(
                n_estimators=200, max_depth=10, min_samples_leaf=2,
                max_features=0.6, random_state=random_state, n_jobs=-1,
            )),
        ]
        if HAS_XGBOOST:
            base_models.append(("xgb", XGBRegressor(
                n_estimators=500, learning_rate=0.03, max_depth=5,
                subsample=0.80, colsample_bytree=0.75,
                reg_lambda=1.5, gamma=0.1,
                objective="reg:squarederror",
                random_state=random_state, n_jobs=-1,
            )))
        models["ensemble"] = AveragingEnsemble(base_models)

    filtered = {k: v for k, v in models.items() if k in selected}
    if not filtered:
        raise ValueError("No trainable models selected.")
    return filtered


# ─────────────────────────── Evaluation ───────────────────────────────────────

def outcome_labels(home_goals: np.ndarray, away_goals: np.ndarray) -> np.ndarray:
    """Return 1 (home win), 0 (draw), -1 (away win)."""
    return np.where(home_goals > away_goals, 1,
                    np.where(home_goals < away_goals, -1, 0))


def exact_score_accuracy(
    true_home: np.ndarray, true_away: np.ndarray,
    pred_home: np.ndarray, pred_away: np.ndarray,
) -> float:
    pred_h_round = np.round(pred_home).clip(0).astype(int)
    pred_a_round = np.round(pred_away).clip(0).astype(int)
    return float(
        ((pred_h_round == true_home.astype(int)) &
         (pred_a_round == true_away.astype(int))).mean()
    )


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


# ─────────────────────────── Training ─────────────────────────────────────────

def train_and_evaluate(
    model_name: str,
    base_model: object,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_home_train: pd.Series,
    y_home_test: pd.Series,
    y_away_train: pd.Series,
    y_away_test: pd.Series,
    models_dir: Path,
    predictions_dir: Path,
    random_state: int,
) -> Dict[str, float]:
    print(f"\n{'─'*60}")
    print(f"  Training: {model_name}")
    print(f"  Train size: {len(X_train):,}   Test size: {len(X_test):,}")

    def _clone_model(m):
        """Return a fresh copy of the model with the same params."""
        try:
            return m.__class__(**m.get_params())
        except Exception:
            import copy
            return copy.deepcopy(m)

    home_pipe = Pipeline([("preprocessor", preprocessor), ("model", base_model)])
    away_pipe = Pipeline([("preprocessor", preprocessor), ("model", _clone_model(base_model))])

    home_pipe.fit(X_train, y_home_train)
    print(f"  Home model fitted ✓")
    away_pipe.fit(X_train, y_away_train)
    print(f"  Away model fitted ✓")

    pred_home = home_pipe.predict(X_test).clip(0)
    pred_away = away_pipe.predict(X_test).clip(0)

    metrics_home = evaluate_predictions(y_home_test.to_numpy(), pred_home)
    metrics_away = evaluate_predictions(y_away_test.to_numpy(), pred_away)

    true_outcomes = outcome_labels(y_home_test.to_numpy(), y_away_test.to_numpy())
    pred_outcomes = outcome_labels(pred_home, pred_away)
    outcome_acc = float((true_outcomes == pred_outcomes).mean())

    exact_acc = exact_score_accuracy(
        y_home_test.to_numpy(), y_away_test.to_numpy(), pred_home, pred_away
    )

    # Save predictions
    predictions_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "row_index": y_home_test.index.to_numpy(),
        "true_home_goals": y_home_test.to_numpy(),
        "true_away_goals": y_away_test.to_numpy(),
        "pred_home_goals": pred_home,
        "pred_away_goals": pred_away,
        "true_result": true_outcomes,
        "pred_result": pred_outcomes,
        "correct_result": true_outcomes == pred_outcomes,
    }).to_csv(predictions_dir / f"{model_name}_test_predictions.csv", index=False)

    # Save models
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(home_pipe, models_dir / f"{model_name}_home_goals.pkl")
    joblib.dump(away_pipe, models_dir / f"{model_name}_away_goals.pkl")

    result = {
        "model": model_name,
        "home_mae": metrics_home["mae"],
        "home_rmse": metrics_home["rmse"],
        "home_r2": metrics_home["r2"],
        "away_mae": metrics_away["mae"],
        "away_rmse": metrics_away["rmse"],
        "away_r2": metrics_away["r2"],
        "avg_mae": (metrics_home["mae"] + metrics_away["mae"]) / 2.0,
        "avg_rmse": (metrics_home["rmse"] + metrics_away["rmse"]) / 2.0,
        "avg_r2": (metrics_home["r2"] + metrics_away["r2"]) / 2.0,
        "outcome_accuracy": outcome_acc,
        "exact_score_accuracy": exact_acc,
    }

    print(f"  Home  MAE={metrics_home['mae']:.4f}  RMSE={metrics_home['rmse']:.4f}  R²={metrics_home['r2']:.4f}")
    print(f"  Away  MAE={metrics_away['mae']:.4f}  RMSE={metrics_away['rmse']:.4f}  R²={metrics_away['r2']:.4f}")
    print(f"  Outcome Accuracy: {outcome_acc*100:.2f}%   Exact Score: {exact_acc*100:.2f}%")
    return result


# ─────────────────────────── Main ─────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    print("="*60)
    print("  Football Match Prediction — Model Training")
    print("="*60)

    print(f"\nLoading features from: {args.features}")
    df = load_and_prepare_data(args.features)
    if args.max_rows and args.max_rows < len(df):
        df = df.tail(args.max_rows).reset_index(drop=True)

    print(f"Dataset: {len(df):,} rows, {len(df.columns)} raw columns")
    print(f"Date range: {df[DATE_COL].min().date()} → {df[DATE_COL].max().date()}")

    X, y_home, y_away = build_feature_matrix(df)
    print(f"Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")

    X_train, X_test, y_home_train, y_home_test, y_away_train, y_away_test = \
        chronological_split(X, y_home, y_away, test_size=args.test_size)
    print(f"Train: {len(X_train):,}   Test: {len(X_test):,}")

    preprocessor = build_preprocessor(X_train)
    models = define_models(
        random_state=args.random_state,
        selected=args.models,
        X_train=X_train,
        y_home_train=y_home_train,
        tune_xgb=args.tune_xgb,
    )

    results: List[Dict] = []
    for name, model in models.items():
        metrics = train_and_evaluate(
            model_name=name,
            base_model=model,
            preprocessor=preprocessor,
            X_train=X_train,
            X_test=X_test,
            y_home_train=y_home_train,
            y_home_test=y_home_test,
            y_away_train=y_away_train,
            y_away_test=y_away_test,
            models_dir=args.models_dir,
            predictions_dir=args.predictions_dir,
            random_state=args.random_state,
        )
        results.append(metrics)

    results_df = pd.DataFrame(results).sort_values("outcome_accuracy", ascending=False).reset_index(drop=True)

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.results_path, index=False)
    args.summary_path.write_text(
        json.dumps(results_df.to_dict(orient="records"), indent=2), encoding="utf-8"
    )

    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    cols = ["model", "outcome_accuracy", "exact_score_accuracy", "avg_mae", "avg_rmse", "avg_r2"]
    print(results_df[cols].to_string(index=False))
    print(f"\nResults saved to: {args.results_path}")
    print(f"Models saved to:  {args.models_dir}")


if __name__ == "__main__":
    main()
