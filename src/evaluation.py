from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def print_report(name: str, y_true, y_pred) -> dict[str, float]:
    metrics = classification_metrics(y_true, y_pred)
    print(f"[{name}] Accuracy: {metrics['accuracy']:.4f}")
    print(f"[{name}] F1-macro: {metrics['f1_macro']:.4f}")
    print(classification_report(y_true, y_pred, digits=4))
    return metrics


def save_confusion_matrix(y_true, y_pred, labels, title: str, output_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
