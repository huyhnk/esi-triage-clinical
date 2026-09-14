"""Train the notebook's numeric + chief-complaint TF-IDF models."""

from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from .config import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, RANDOM_STATE, RESULTS_DIR, TEST_SIZE, TOP10_NUMERIC_FEATURES
from .evaluation import print_report, save_confusion_matrix


def to_three_classes(y: pd.Series) -> pd.Series:
    y = y.astype(int).copy()
    y.loc[y >= 3] = 3
    return y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PROCESSED_DIR / "master_dataset_clean.csv")
    parser.add_argument("--max-features", type=int, default=3000)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    required = TOP10_NUMERIC_FEATURES + ["acuity", "chiefcomplaint"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    X_num = df[TOP10_NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    X_text = df["chiefcomplaint"].fillna("").astype(str)
    y = to_three_classes(df["acuity"])

    indices = df.index.to_numpy()
    idx_train, idx_test, y_train, y_test = train_test_split(
        indices, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    num_imputer = SimpleImputer(strategy="median")
    X_num_train = num_imputer.fit_transform(X_num.loc[idx_train])
    X_num_test = num_imputer.transform(X_num.loc[idx_test])

    tfidf = TfidfVectorizer(max_features=args.max_features, ngram_range=(1, 2), min_df=5)
    X_text_train = tfidf.fit_transform(X_text.loc[idx_train])
    X_text_test = tfidf.transform(X_text.loc[idx_test])

    X_train = hstack([csr_matrix(X_num_train), X_text_train]).tocsr()
    X_test = hstack([csr_matrix(X_num_test), X_text_test]).tocsr()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    logit = LogisticRegression(max_iter=300, solver="saga", n_jobs=-1, random_state=RANDOM_STATE)
    logit.fit(X_train, y_train)
    pred_logit = logit.predict(X_test)
    logit_metrics = print_report("Logistic-num+TFIDF", y_test, pred_logit)

    xgb = XGBClassifier(
        n_estimators=250, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
        tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1,
    )
    xgb.fit(X_train, y_train - 1)
    pred_xgb = xgb.predict(X_test) + 1
    xgb_metrics = print_report("XGBoost-num+TFIDF", y_test, pred_xgb)
    save_confusion_matrix(y_test, pred_xgb, [1, 2, 3], "XGBoost - numeric + TF-IDF", FIGURES_DIR / "cm_xgb_numeric_tfidf.png")

    preprocess = {"numeric_imputer": num_imputer, "tfidf": tfidf, "numeric_features": TOP10_NUMERIC_FEATURES}
    joblib.dump(preprocess, MODELS_DIR / "preprocess_numeric_tfidf.joblib")
    joblib.dump(logit, MODELS_DIR / "logistic_numeric_tfidf.joblib")
    joblib.dump(xgb, MODELS_DIR / "xgboost_numeric_tfidf.joblib")

    results = pd.DataFrame([
        {"model": "LogisticRegression", "features": f"top10_numeric + TFIDF({args.max_features})", "text": "yes", "n_classes": 3, **logit_metrics},
        {"model": "XGBoost", "features": f"top10_numeric + TFIDF({args.max_features})", "text": "yes", "n_classes": 3, **xgb_metrics},
    ])
    results.to_csv(RESULTS_DIR / "multimodal_3class_results.csv", index=False)
    print(results)


if __name__ == "__main__":
    main()
