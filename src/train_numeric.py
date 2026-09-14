"""Train the final 3-class numeric baselines from the notebook."""

from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .config import MODELS_DIR, PROCESSED_DIR, RANDOM_STATE, RESULTS_DIR, FIGURES_DIR, TEST_SIZE, TOP10_NUMERIC_FEATURES
from .evaluation import print_report, save_confusion_matrix


def to_three_classes(y: pd.Series) -> pd.Series:
    y = y.astype(int).copy()
    y.loc[y >= 3] = 3
    return y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PROCESSED_DIR / "master_dataset_clean.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    missing = [c for c in TOP10_NUMERIC_FEATURES + ["acuity"] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    X = df[TOP10_NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    y = to_three_classes(df["acuity"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(n_estimators=300, max_features="sqrt", n_jobs=-1, random_state=RANDOM_STATE)),
    ])
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    rf_metrics = print_report("RandomForest-3class", y_test, pred_rf)
    save_confusion_matrix(y_test, pred_rf, [1, 2, 3], "Random Forest - 3 class", FIGURES_DIR / "cm_rf_3class.png")
    joblib.dump(rf, MODELS_DIR / "random_forest_3class.joblib")

    # Use training medians for a leakage-safe imputation before XGBoost.
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
        tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1,
    )
    xgb.fit(X_train_imp, y_train - 1)
    pred_xgb = xgb.predict(X_test_imp) + 1
    xgb_metrics = print_report("XGBoost-3class", y_test, pred_xgb)
    save_confusion_matrix(y_test, pred_xgb, [1, 2, 3], "XGBoost - 3 class", FIGURES_DIR / "cm_xgb_3class.png")
    joblib.dump({"imputer": imputer, "model": xgb}, MODELS_DIR / "xgboost_3class.joblib")

    results = pd.DataFrame([
        {"model": "RandomForest", "features": "top10_numeric", "n_classes": 3, **rf_metrics},
        {"model": "XGBoost", "features": "top10_numeric", "n_classes": 3, **xgb_metrics},
    ])
    results.to_csv(RESULTS_DIR / "numeric_3class_results.csv", index=False)
    print(results)


if __name__ == "__main__":
    main()
