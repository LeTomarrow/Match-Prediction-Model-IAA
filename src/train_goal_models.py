#!/usr/bin/env python3
"""Train football match prediction models.

Improvements:
  - Extra features from raw CSVs (Elo, streaks, days rest, position, competition context)
  - Poisson outcome decoder with Dixon-Coles correction
  - 3-class XGBoost classifier
  - Probability blending with calibrated draw boost
"""
from __future__ import annotations
import argparse, copy, json, sys, warnings
from math import exp, lgamma, log
from pathlib import Path
from typing import Dict, List, Tuple
import joblib, numpy as np, pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("WARNING: xgboost not installed.")

TARGET_HOME = "home_club_goals"
TARGET_AWAY = "away_club_goals"
DATE_COL = "date"
ID_COLS = {"game_id", "home_club_id", "away_club_id",
           "home_feature_date", "away_feature_date", TARGET_HOME, TARGET_AWAY}
OUTCOME_MAP = {1: "Home Win", 0: "Draw", -1: "Away Win"}

# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path,
                   default=Path(__file__).resolve().parents[1] / "data" / "match_features.csv")
    p.add_argument("--data-dir", type=Path,
                   default=Path(__file__).resolve().parents[1] / "data")
    p.add_argument("--models-dir", type=Path,
                   default=Path(__file__).resolve().parents[1] / "models")
    p.add_argument("--results-path", type=Path,
                   default=Path(__file__).resolve().parents[1] / "data" / "model_results.csv")
    p.add_argument("--summary-path", type=Path,
                   default=Path(__file__).resolve().parents[1] / "data" / "model_results_summary.json")
    p.add_argument("--predictions-dir", type=Path,
                   default=Path(__file__).resolve().parents[1] / "data" / "predictions")
    p.add_argument("--test-size", type=float, default=0.20)
    p.add_argument("--val-size", type=float, default=0.10)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--models", nargs="+",
                   choices=["linear_regression", "random_forest", "xgboost"],
                   default=["linear_regression", "random_forest", "xgboost"])
    p.add_argument("--skip-extra-features", action="store_true", default=False)
    return p.parse_args()

# ── Poisson / Dixon-Coles ─────────────────────────────────────────────────────
def _pmf(k: int, lam: float) -> float:
    lam = max(lam, 1e-9)
    return exp(k * log(lam) - lam - lgamma(k + 1))

def _dc_tau(h, a, lh, la, rho):
    if h == 0 and a == 0: return 1 - lh * la * rho
    if h == 1 and a == 0: return 1 + la * rho
    if h == 0 and a == 1: return 1 + lh * rho
    if h == 1 and a == 1: return 1 - rho
    return 1.0

def poisson_outcome_probs(lh, la, rho=-0.15, max_g=8):
    lh, la = max(lh, 0.05), max(la, 0.05)
    pw = pd_ = pa = 0.0
    for h in range(max_g + 1):
        for a in range(max_g + 1):
            p = _pmf(h, lh) * _pmf(a, la) * _dc_tau(h, a, lh, la, rho)
            if h > a:    pw  += p
            elif h == a: pd_ += p
            else:        pa  += p
    tot = pw + pd_ + pa
    return (pw/tot, pd_/tot, pa/tot) if tot > 0 else (1/3, 1/3, 1/3)

def poisson_proba_batch(ph, pa, rho=-0.15):
    """Return (N,3) array [P_away, P_draw, P_home]."""
    out = np.zeros((len(ph), 3))
    for i, (lh, la) in enumerate(zip(ph, pa)):
        pw, pd_, pa_ = poisson_outcome_probs(float(lh), float(la), rho)
        out[i] = [pa_, pd_, pw]
    return out

def proba_to_outcome(proba):
    idx = proba.argmax(axis=1)
    return np.where(idx == 2, 1, np.where(idx == 1, 0, -1))

def calibrate_blend(val_poisson, val_clf, val_true):
    """Grid-search (w_clf, draw_boost) on validation. Returns (params, best_acc)."""
    best_acc, best_p = 0.0, (0.4, 1.0)
    w_options = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0] if val_clf is not None else [0.0]
    for w in w_options:
        for db in [1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0]:
            if val_clf is not None:
                blended = (1 - w) * val_poisson + w * val_clf
            else:
                blended = val_poisson.copy()
            blended[:, 1] *= db
            blended /= blended.sum(axis=1, keepdims=True)
            acc = float((proba_to_outcome(blended) == val_true).mean())
            if acc > best_acc:
                best_acc, best_p = acc, (w, db)
    return best_p, best_acc

# ── Averaging ensemble ────────────────────────────────────────────────────────
class AveragingEnsemble(BaseEstimator):
    def __init__(self, estimators):
        self.estimators = estimators
    def fit(self, X, y):
        self.fitted_ = [(n, copy.deepcopy(e).fit(X, y)) for n, e in self.estimators]
        return self
    def predict(self, X):
        return np.mean([e.predict(X) for _, e in self.fitted_], axis=0)

# ── Feature engineering ───────────────────────────────────────────────────────
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pairs = [
        ("team_ppg_3","ppg3"),("team_ppg_5","ppg5"),("team_ppg_10","ppg10"),
        ("team_goal_diff_avg_3","gd3"),("team_goal_diff_avg_5","gd5"),
        ("team_goal_diff_avg_10","gd10"),
        ("team_goals_scored_avg_3","gs3"),("team_goals_scored_avg_5","gs5"),
        ("team_goals_scored_avg_10","gs10"),
        ("team_goals_conceded_avg_3","gc3"),("team_goals_conceded_avg_5","gc5"),
        ("team_goals_conceded_avg_10","gc10"),
        ("team_win_rate_3","wr3"),("team_win_rate_5","wr5"),("team_win_rate_10","wr10"),
        ("team_form_weighted_points_3","fwp3"),("team_form_weighted_points_5","fwp5"),
        ("squad_total_market_value_eur","squad_mv"),
        ("squad_top11_market_value_eur","top11_mv"),
        ("squad_value_per_starter_eur","starter_mv"),
        ("win_streak","wstreak"),("loss_streak","lstreak"),
        ("unbeaten_streak","ustreak"),("days_rest","rest"),
        ("league_position","pos"),
    ]
    for sfx, short in pairs:
        hc, ac = f"home_{sfx}", f"away_{sfx}"
        if hc in df.columns and ac in df.columns:
            df[f"diff_{short}"] = df[hc] - df[ac]
            df[f"ratio_{short}"] = df[hc] / df[ac].replace(0, np.nan)

    for w in (3, 5, 10):
        hgs = f"home_team_goals_scored_avg_{w}"; agc = f"away_team_goals_conceded_avg_{w}"
        ags = f"away_team_goals_scored_avg_{w}"; hgc = f"home_team_goals_conceded_avg_{w}"
        if all(c in df.columns for c in [hgs, agc, ags, hgc]):
            df[f"home_attack_vs_away_def_{w}"] = df[hgs] - df[agc]
            df[f"away_attack_vs_home_def_{w}"] = df[ags] - df[hgc]

    for side in ("home", "away"):
        p3,p10 = f"{side}_team_ppg_3",f"{side}_team_ppg_10"
        g3,g10 = f"{side}_team_goal_diff_avg_3",f"{side}_team_goal_diff_avg_10"
        if p3 in df.columns and p10 in df.columns:
            df[f"{side}_momentum_ppg"] = df[p3] - df[p10]
        if g3 in df.columns and g10 in df.columns:
            df[f"{side}_momentum_gd"] = df[g3] - df[g10]

    if all(c in df.columns for c in ["home_team_goals_scored_avg_5","away_team_goals_conceded_avg_5",
                                      "away_team_goals_scored_avg_5","home_team_goals_conceded_avg_5"]):
        mu = 1.35
        ah = (df["home_team_goals_scored_avg_5"]/mu).clip(0.1)
        da = (df["away_team_goals_conceded_avg_5"]/mu).clip(0.1)
        aa = (df["away_team_goals_scored_avg_5"]/mu).clip(0.1)
        dh = (df["home_team_goals_conceded_avg_5"]/mu).clip(0.1)
        df["prior_lambda_home"] = ah * da * mu * 1.15
        df["prior_lambda_away"] = aa * dh * mu

    if "home_elo" in df.columns and "away_elo" in df.columns:
        df["elo_ratio"] = df["home_elo"] / df["away_elo"].replace(0, np.nan)

    return df

# ── Data loading ──────────────────────────────────────────────────────────────
def load_data(features_path, data_dir, skip_extra):
    print(f"Loading {features_path.name} ...")
    df = pd.read_csv(features_path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, TARGET_HOME, TARGET_AWAY]).copy()
    sc = [DATE_COL, "game_id"] if "game_id" in df.columns else [DATE_COL]
    df = df.sort_values(sc).reset_index(drop=True)

    if not skip_extra:
        try:
            src_dir = str(Path(__file__).resolve().parent)
            if src_dir not in sys.path:
                sys.path.insert(0, src_dir)
            from feature_extras import build_extra_features
            extra = build_extra_features(data_dir)
            before = len(df.columns)
            df = df.merge(extra, on="game_id", how="left")
            print(f"  Extra features merged: {len(df.columns) - before} new columns")
        except Exception as e:
            print(f"  WARNING: Could not load extra features: {e}")

    return df

def build_feature_matrix(df):
    df = add_engineered_features(df)
    drop = ID_COLS | {DATE_COL, "home_feature_date", "away_feature_date"}
    X = df[[c for c in df.columns if c not in drop]].copy()
    yh = pd.to_numeric(df[TARGET_HOME], errors="coerce")
    ya = pd.to_numeric(df[TARGET_AWAY], errors="coerce")
    valid = yh.notna() & ya.notna()
    return (X.loc[valid].reset_index(drop=True),
            yh.loc[valid].reset_index(drop=True),
            ya.loc[valid].reset_index(drop=True))

# ── Splits ────────────────────────────────────────────────────────────────────
def three_way_split(X, yh, ya, val_size, test_size):
    n = len(X)
    iv = int(n * (1 - val_size - test_size))
    it = int(n * (1 - test_size))
    return (X.iloc[:iv], X.iloc[iv:it], X.iloc[it:],
            yh.iloc[:iv], yh.iloc[iv:it], yh.iloc[it:],
            ya.iloc[:iv], ya.iloc[iv:it], ya.iloc[it:])

# ── Preprocessing ─────────────────────────────────────────────────────────────
def build_preprocessor(X):
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    trs = []
    if num:
        trs.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                     ("sc", StandardScaler())]), num))
    if cat:
        trs.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                     ("enc", OneHotEncoder(handle_unknown="ignore",
                                                           sparse_output=False))]), cat))
    return ColumnTransformer(trs, remainder="drop")

# ── Outcome helpers ───────────────────────────────────────────────────────────
def outcome_labels(h, a):
    return np.where(h > a, 1, np.where(h < a, -1, 0))

def per_class_acc(true, pred):
    out = {}
    for label, name in OUTCOME_MAP.items():
        mask = true == label
        out[name] = float((pred[mask] == label).mean()) if mask.sum() > 0 else float("nan")
        out[f"n_{name}"] = int(mask.sum())
    return out

# ── Model definitions ─────────────────────────────────────────────────────────
def get_models(selected, rs):
    m = {}
    if "linear_regression" in selected:
        m["linear_regression"] = LinearRegression()
    if "random_forest" in selected:
        m["random_forest"] = RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=2,
            max_features=0.6, random_state=rs, n_jobs=-1)
    if "xgboost" in selected and HAS_XGBOOST:
        m["xgboost"] = XGBRegressor(
            n_estimators=700, learning_rate=0.03, max_depth=5,
            min_child_weight=3, subsample=0.80, colsample_bytree=0.75,
            reg_alpha=0.1, reg_lambda=1.5, gamma=0.1,
            objective="reg:squarederror", random_state=rs, n_jobs=-1)
    return m

# ── 3-class classifier ────────────────────────────────────────────────────────
def train_classifier(prep, X_tr, yh_tr, ya_tr, rs):
    if not HAS_XGBOOST:
        return None
    y_cls = outcome_labels(yh_tr.to_numpy(), ya_tr.to_numpy())
    label_map = {1: 2, 0: 1, -1: 0}
    y_enc = np.array([label_map[v] for v in y_cls])
    clf = Pipeline([
        ("prep", prep),
        ("clf", XGBClassifier(
            n_estimators=500, learning_rate=0.03, max_depth=5,
            subsample=0.80, colsample_bytree=0.75, reg_lambda=1.5,
            eval_metric="mlogloss", random_state=rs, n_jobs=-1)),
    ])
    clf.fit(X_tr, y_enc)
    return clf

def clf_proba(clf, X):
    """(N,3) proba [away, draw, home]."""
    if clf is None:
        return None
    return clf.predict_proba(X)   # XGB classes: 0=AW, 1=D, 2=HW

# ── Train & evaluate ──────────────────────────────────────────────────────────
def train_and_evaluate(name, model, prep, X_tr, X_val, X_te,
                       yh_tr, yh_val, yh_te, ya_tr, ya_val, ya_te,
                       models_dir, preds_dir, rs):
    print(f"\n{'─'*60}\n  {name}  "
          f"train={len(X_tr):,}  val={len(X_val):,}  test={len(X_te):,}")

    home_pipe = Pipeline([("prep", prep), ("m", model)])
    away_pipe = Pipeline([("prep", prep), ("m", copy.deepcopy(model))])
    home_pipe.fit(X_tr, yh_tr); print("  home regressor fitted ✓")
    away_pipe.fit(X_tr, ya_tr); print("  away regressor fitted ✓")

    # ── Validation: calibrate blend ───────────────────────────────────────────
    val_ph = home_pipe.predict(X_val).clip(0)
    val_pa = away_pipe.predict(X_val).clip(0)
    val_true = outcome_labels(yh_val.to_numpy(), ya_val.to_numpy())

    val_pois_p = poisson_proba_batch(val_ph, val_pa)

    clf = train_classifier(prep, X_tr, yh_tr, ya_tr, rs)
    val_clf_p = clf_proba(clf, X_val)
    if val_clf_p is not None:
        print(f"  Classifier val acc (raw): "
              f"{accuracy_score(val_true, proba_to_outcome(val_clf_p))*100:.2f}%")

    (best_w, best_db), best_val = calibrate_blend(val_pois_p, val_clf_p, val_true)
    print(f"  Calibrated: w_clf={best_w:.2f}  draw_boost={best_db:.2f}  "
          f"val_acc={best_val*100:.2f}%")

    # ── Test ──────────────────────────────────────────────────────────────────
    pred_h = home_pipe.predict(X_te).clip(0)
    pred_a = away_pipe.predict(X_te).clip(0)
    true_out = outcome_labels(yh_te.to_numpy(), ya_te.to_numpy())

    te_pois_p = poisson_proba_batch(pred_h, pred_a)
    te_clf_p  = clf_proba(clf, X_te)

    if te_clf_p is not None:
        blended = (1 - best_w) * te_pois_p + best_w * te_clf_p
    else:
        blended = te_pois_p.copy()
    blended[:, 1] *= best_db
    blended /= blended.sum(axis=1, keepdims=True)
    blended_out = proba_to_outcome(blended)
    blended_acc = float((blended_out == true_out).mean())

    pois_acc = float((proba_to_outcome(te_pois_p) == true_out).mean())
    clf_acc  = float((proba_to_outcome(te_clf_p) == true_out).mean()) if te_clf_p is not None else 0.0

    pc = per_class_acc(true_out, blended_out)
    mh = dict(mae=mean_absolute_error(yh_te, pred_h),
               rmse=float(np.sqrt(mean_squared_error(yh_te, pred_h))),
               r2=r2_score(yh_te, pred_h))
    ma = dict(mae=mean_absolute_error(ya_te, pred_a),
               rmse=float(np.sqrt(mean_squared_error(ya_te, pred_a))),
               r2=r2_score(ya_te, pred_a))
    exact = float(((np.round(pred_h).clip(0).astype(int) == yh_te.astype(int)) &
                   (np.round(pred_a).clip(0).astype(int) == ya_te.astype(int))).mean())

    print(f"  Poisson-only  acc: {pois_acc*100:.2f}%")
    if te_clf_p is not None:
        print(f"  Classifier-only acc: {clf_acc*100:.2f}%")
    print(f"  Blended acc  : {blended_acc*100:.2f}%  <- FINAL")
    print(f"  Per-class  Home={pc.get('Home Win',0)*100:.1f}%  "
          f"Draw={pc.get('Draw',0)*100:.1f}%  "
          f"Away={pc.get('Away Win',0)*100:.1f}%")
    print(f"  Home MAE={mh['mae']:.4f}  Away MAE={ma['mae']:.4f}")

    # Save
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(home_pipe, models_dir / f"{name}_home_goals.pkl")
    joblib.dump(away_pipe, models_dir / f"{name}_away_goals.pkl")
    if clf:
        joblib.dump(clf, models_dir / f"{name}_outcome_classifier.pkl")

    preds_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "true_home_goals": yh_te.to_numpy(), "true_away_goals": ya_te.to_numpy(),
        "pred_home_goals": pred_h, "pred_away_goals": pred_a,
        "true_result": true_out,
        "poisson_result": proba_to_outcome(te_pois_p),
        "clf_result": proba_to_outcome(te_clf_p) if te_clf_p is not None else np.nan,
        "pred_result": blended_out,
        "correct_result": blended_out == true_out,
    }).to_csv(preds_dir / f"{name}_test_predictions.csv", index=False)

    return {
        "model": name,
        "outcome_accuracy": blended_acc,
        "poisson_accuracy": pois_acc,
        "classifier_accuracy": clf_acc,
        "home_win_accuracy": pc.get("Home Win", float("nan")),
        "draw_accuracy": pc.get("Draw", float("nan")),
        "away_win_accuracy": pc.get("Away Win", float("nan")),
        "exact_score_accuracy": exact,
        "best_w_clf": best_w, "best_draw_boost": best_db,
        "home_mae": mh["mae"], "home_rmse": mh["rmse"], "home_r2": mh["r2"],
        "away_mae": ma["mae"], "away_rmse": ma["rmse"], "away_r2": ma["r2"],
        "avg_mae": (mh["mae"] + ma["mae"]) / 2,
        "avg_rmse": (mh["rmse"] + ma["rmse"]) / 2,
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    print("=" * 60)
    print("  Football Match Prediction — Enhanced Training")
    print("=" * 60)

    df = load_data(args.features, args.data_dir, args.skip_extra_features)
    print(f"Dataset: {len(df):,} rows | "
          f"{df[DATE_COL].min().date()} -> {df[DATE_COL].max().date()}")

    X, yh, ya = build_feature_matrix(df)
    print(f"Feature matrix: {X.shape[0]:,} x {X.shape[1]}")

    (X_tr, X_val, X_te,
     yh_tr, yh_val, yh_te,
     ya_tr, ya_val, ya_te) = three_way_split(X, yh, ya, args.val_size, args.test_size)

    prep = build_preprocessor(X_tr)
    models = get_models(args.models, args.random_state)

    results = []
    for name, model in models.items():
        r = train_and_evaluate(
            name, model, prep,
            X_tr, X_val, X_te,
            yh_tr, yh_val, yh_te,
            ya_tr, ya_val, ya_te,
            args.models_dir, args.predictions_dir, args.random_state)
        results.append(r)

    rdf = pd.DataFrame(results).sort_values("outcome_accuracy", ascending=False).reset_index(drop=True)
    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    rdf.to_csv(args.results_path, index=False)
    args.summary_path.write_text(json.dumps(rdf.to_dict(orient="records"), indent=2))

    print("\n" + "=" * 60 + "\n  FINAL SUMMARY\n" + "=" * 60)
    cols = ["model", "outcome_accuracy", "draw_accuracy",
            "home_win_accuracy", "away_win_accuracy", "exact_score_accuracy"]
    print(rdf[cols].to_string(index=False))
    print(f"\nResults -> {args.results_path}")

if __name__ == "__main__":
    main()
