#!/usr/bin/env python3
"""
Train models following the paper-style experiments using `data/paper_match_features.csv`.

Trains Logistic Regression, Random Forest, SVM, XGBoost (if available), LightGBM (if available),
and a stacking ensemble. Evaluates accuracy, precision, recall, F1, and MCC.

Outputs:
 - `data/paper_predictions.csv` with test predictions and probabilities
 - `models/paper_models.pkl` (joblib dict of trained estimators)
 - prints metrics summary
"""
from __future__ import annotations

from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None


def load_features(data_dir: Path) -> pd.DataFrame:
    p = data_dir / 'paper_match_features.csv'
    if not p.exists():
        raise FileNotFoundError(f'{p} not found — run src/build_paper_features.py first')
    df = pd.read_csv(p)
    return df


def split_train_test(df: pd.DataFrame):
    # If season column present in original games.csv, prefer season split
    games = pd.read_csv(Path('data') / 'games.csv')
    if 'season' in games.columns:
        seasons = games[['game_id', 'season']]
        df = df.merge(seasons, on='game_id', how='left')
        # default: train seasons 2019,2020,2021 test 2022
        train_seasons = [2019, 2020, 2021]
        test_seasons = [2022]
        train = df[df['season'].isin(train_seasons)].copy()
        test = df[df['season'].isin(test_seasons)].copy()
        if len(test) == 0:
            # fallback: temporal split
            df = df.sort_values('date')
            cutoff = int(len(df) * 0.8)
            train = df.iloc[:cutoff].copy()
            test = df.iloc[cutoff:].copy()
    else:
        df = df.sort_values('date')
        cutoff = int(len(df) * 0.8)
        train = df.iloc[:cutoff].copy()
        test = df.iloc[cutoff:].copy()
    return train, test


def get_feature_columns(df: pd.DataFrame) -> list:
    drop = ['game_id', 'date', 'home_club_goals', 'away_club_goals', 'target']
    return [c for c in df.columns if c not in drop]


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    return ColumnTransformer(
        transformers=[
            (
                'num',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                ]),
                numeric_cols,
            ),
            (
                'cat',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore')),
                ]),
                categorical_cols,
            ),
        ],
        remainder='drop',
    )


def make_model_pipeline(X: pd.DataFrame, estimator) -> Pipeline:
    return Pipeline([
        ('preprocess', build_preprocessor(X)),
        ('clf', estimator),
    ])


def train_and_eval(train: pd.DataFrame, test: pd.DataFrame, data_dir: Path):
    X_train = train[get_feature_columns(train)]
    y_train = train['target']
    X_test = test[get_feature_columns(test)]
    y_test = test['target']

    preprocessor = build_preprocessor(X_train)

    models = {}

    # Logistic Regression
    lr = make_model_pipeline(X_train, LogisticRegression(max_iter=1000))
    lr.fit(X_train, y_train)
    models['logistic'] = lr

    # Random Forest
    rf = make_model_pipeline(X_train, RandomForestClassifier(n_estimators=200, random_state=0))
    rf.fit(X_train, y_train)
    models['random_forest'] = rf

    # SVM (probabilities)
    svc = make_model_pipeline(X_train, SVC(probability=True, kernel='rbf'))
    svc.fit(X_train, y_train)
    models['svc'] = svc

    # XGBoost if available
    if XGBClassifier is not None:
        xgb = make_model_pipeline(X_train, XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
        xgb.fit(X_train, y_train)
        models['xgboost'] = xgb

    # LightGBM if available
    if LGBMClassifier is not None:
        lgbm = make_model_pipeline(X_train, LGBMClassifier())
        lgbm.fit(X_train, y_train)
        models['lightgbm'] = lgbm

    # Stacking ensemble using LR meta-learner
    estimators = [(name, est) for name, est in models.items() if name in ['random_forest', 'xgboost', 'lightgbm', 'svc']]
    if not estimators:
        estimators = [('rf', models['random_forest']), ('lr', models['logistic'])]

    stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=5, n_jobs=-1, passthrough=False)
    stack.fit(X_train, y_train)
    models['stacking'] = stack

    # Evaluate
    results = {}
    preds = []
    for name, model in models.items():
        prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)
        yhat = (prob >= 0.5).astype(int)
        results[name] = {
            'accuracy': float(accuracy_score(y_test, yhat)),
            'precision': float(precision_score(y_test, yhat, zero_division=0)),
            'recall': float(recall_score(y_test, yhat, zero_division=0)),
            'f1': float(f1_score(y_test, yhat, zero_division=0)),
            'mcc': float(matthews_corrcoef(y_test, yhat)),
            'confusion_matrix': confusion_matrix(y_test, yhat).tolist()
        }

        preds.append((name, prob, yhat))

    # Save predictions for stacking (each model prob)
    pred_df = test[['game_id', 'date', 'home_club_id', 'away_club_id', 'target']].copy()
    for name, prob, yhat in preds:
        pred_df[f'prob_{name}'] = prob
        pred_df[f'pred_{name}'] = yhat

    pred_out = data_dir / 'paper_predictions.csv'
    pred_df.to_csv(pred_out, index=False)

    # Save models
    joblib.dump(models, Path('models') / 'paper_models.pkl')

    # Save results summary
    with open(data_dir / 'paper_model_results.json', 'w') as fh:
        json.dump(results, fh, indent=2)

    print('Model results:')
    for name, r in results.items():
        print(f"{name}: acc={r['accuracy']:.3f} prec={r['precision']:.3f} rec={r['recall']:.3f} f1={r['f1']:.3f} mcc={r['mcc']:.3f}")


def main():
    data_dir = Path('data')
    df = load_features(data_dir)
    train, test = split_train_test(df)
    if len(test) == 0:
        raise RuntimeError('Test set is empty — adjust split or seasons')
    Path('models').mkdir(exist_ok=True)
    train_and_eval(train, test, data_dir)


if __name__ == '__main__':
    main()
