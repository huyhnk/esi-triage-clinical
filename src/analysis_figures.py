"""Generate the correlation and Random-Forest feature-importance figures from the notebook."""

from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from .config import FIGURES_DIR, PROCESSED_DIR, RANDOM_STATE, TOP10_NUMERIC_FEATURES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PROCESSED_DIR / "master_dataset_clean.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    corr_df = df[TOP10_NUMERIC_FEATURES + ["acuity"]].apply(pd.to_numeric, errors="coerce")
    corr_df = corr_df.fillna(corr_df.median(numeric_only=True))
    corr = corr_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(corr.values, aspect="auto")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.index)), corr.index)
    ax.set_title("Correlation matrix: numeric features and acuity")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_numeric_acuity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    X = df[TOP10_NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    y = df["acuity"].astype(int).copy()
    y.loc[y >= 3] = 3
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    imp = SimpleImputer(strategy="median")
    X_train = imp.fit_transform(X_train)
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    importance_df = pd.DataFrame({"feature": TOP10_NUMERIC_FEATURES, "importance": rf.feature_importances_}).sort_values("importance")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(importance_df["feature"], importance_df["importance"])
    ax.set_title("Random Forest feature importance")
    ax.set_xlabel("Feature importance")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance_random_forest.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
