"""
classify.py
-----------
Machine learning classification of EEG recordings by genotype (WT vs KO).

Takes per-epoch discharge summaries produced by detection.process_group()
and trains a Random Forest classifier to distinguish wild-type (WT) from
knockout (KO) animals based on seizure-associated discharge features.

Pipeline
--------
    1. Feature engineering from discharge summaries + band power
    2. Train / test split stratified by genotype
    3. Random Forest classifier with cross-validation
    4. Evaluation: ROC-AUC, precision-recall, confusion matrix
    5. Feature importance (which EEG features drive the classification)

Typical usage
-------------
    import pandas as pd
    from classify import build_feature_matrix, train_classifier, evaluate_classifier

    df_wt = pd.read_csv("results/wt_summaries.csv")
    df_ko = pd.read_csv("results/ko_summaries.csv")

    X, y = build_feature_matrix(df_wt, df_ko)
    model, X_test, y_test = train_classifier(X, y)
    metrics = evaluate_classifier(model, X_test, y_test)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "n_events",
    "rate_per_min",
    "mean_voltage_mV",
    "std_voltage_mV",
    "mean_prominence",
]

BAND_COLS = ["delta", "theta", "alpha", "beta", "gamma"]


def build_feature_matrix(
    df_wt: pd.DataFrame,
    df_ko: pd.DataFrame,
    include_bands: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Combine WT and KO summary DataFrames into a labelled feature matrix.

    Parameters
    ----------
    df_wt : pd.DataFrame
        Per-epoch summaries for wild-type group (from detection.process_group).
    df_ko : pd.DataFrame
        Per-epoch summaries for knockout group (from detection.process_group).
    include_bands : bool
        If True and band power columns are present, include them as features.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix — one row per epoch.
    y : np.ndarray
        Integer labels: 0 = WT, 1 = KO.

    Notes
    -----
    Labels are intentionally 0/1 rather than strings so they are compatible
    with all sklearn estimators without further encoding.
    """
    df_wt = df_wt.copy()
    df_ko = df_ko.copy()
    df_wt["label"] = 0
    df_ko["label"] = 1

    df_all = pd.concat([df_wt, df_ko], ignore_index=True)

    feature_cols = [c for c in FEATURE_COLS if c in df_all.columns]
    if include_bands:
        feature_cols += [c for c in BAND_COLS if c in df_all.columns]

    X = df_all[feature_cols].copy()
    y = df_all["label"].values

    return X, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_classifier(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = 0.25,
    n_estimators: int = 200,
    random_state: int = 42,
    cv_folds: int = 5,
) -> tuple[Pipeline, pd.DataFrame, np.ndarray]:
    """
    Train a Random Forest classifier with cross-validation.

    A sklearn Pipeline is used so that scaling and classification are
    treated as a single reproducible unit — there is no data leakage
    between train and test splits.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix from build_feature_matrix().
    y : np.ndarray
        Integer labels (0 = WT, 1 = KO).
    test_size : float
        Proportion of data held out for final evaluation (default 0.25).
    n_estimators : int
        Number of trees in the Random Forest (default 200).
    random_state : int
        Random seed for reproducibility (default 42).
    cv_folds : int
        Number of stratified cross-validation folds (default 5).

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        Fitted pipeline (StandardScaler + RandomForestClassifier).
    X_test : pd.DataFrame
        Held-out feature matrix for evaluation.
    y_test : np.ndarray
        Held-out labels for evaluation.

    Prints
    ------
    Cross-validation ROC-AUC scores (mean ± std) on training set.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )),
    ])

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")

    print(f"Cross-validation ROC-AUC ({cv_folds}-fold): "
          f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    pipeline.fit(X_train, y_train)

    return pipeline, X_test, y_test


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_classifier(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    label_names: list[str] | None = None,
) -> dict:
    """
    Evaluate a fitted classifier on the held-out test set.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted sklearn Pipeline from train_classifier().
    X_test : pd.DataFrame
        Held-out feature matrix.
    y_test : np.ndarray
        Held-out labels.
    label_names : list of str, optional
        Human-readable class names (default ["WT", "KO"]).

    Returns
    -------
    metrics : dict
        Keys:
            'roc_auc'           : float — area under ROC curve
            'average_precision' : float — area under precision-recall curve
            'classification_report' : str — sklearn classification report
            'confusion_matrix'  : np.ndarray — 2×2 confusion matrix
            'fpr', 'tpr'        : arrays for plotting ROC curve
            'precision', 'recall' : arrays for plotting PR curve
    """
    if label_names is None:
        label_names = ["WT", "KO"]

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "average_precision": average_precision_score(y_test, y_prob),
        "classification_report": classification_report(
            y_test, y_pred, target_names=label_names
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
    }

    print(f"\nTest ROC-AUC : {metrics['roc_auc']:.3f}")
    print(f"Avg Precision: {metrics['average_precision']:.3f}")
    print("\nClassification report:")
    print(metrics["classification_report"])

    return metrics


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(
    pipeline: Pipeline,
    feature_names: list[str],
    save_path: str | None = None,
) -> None:
    """
    Plot and optionally save a feature importance bar chart.

    Feature importances are extracted from the Random Forest and sorted
    in descending order. This reveals which EEG features (e.g. gamma power,
    discharge rate) are most discriminative between WT and KO genotypes —
    providing biological interpretability beyond classification accuracy.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted pipeline containing a RandomForestClassifier step named 'clf'.
    feature_names : list of str
        Names of features in the same order as columns of X.
    save_path : str, optional
        If provided, saves the figure to this path (e.g. "figures/importance.png").
    """
    rf = pipeline.named_steps["clf"]
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)

    idx = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in idx]
    sorted_imp = importances[idx]
    sorted_std = std[idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        sorted_names[::-1],
        sorted_imp[::-1],
        xerr=sorted_std[::-1],
        color="#1D9E75",
        alpha=0.85,
        capsize=3,
    )
    ax.set_xlabel("Feature importance (mean decrease in impurity)", fontsize=11)
    ax.set_title("Which EEG features distinguish WT from KO?", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Evaluation plots
# ---------------------------------------------------------------------------

def plot_roc_and_pr(
    metrics: dict,
    save_path: str | None = None,
) -> None:
    """
    Plot ROC curve and Precision-Recall curve side by side.

    Parameters
    ----------
    metrics : dict
        Output of evaluate_classifier().
    save_path : str, optional
        If provided, saves the figure (e.g. "figures/roc_pr.png").
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ROC
    ax = axes[0]
    ax.plot(
        metrics["fpr"], metrics["tpr"],
        color="#1D9E75", lw=2,
        label=f"AUC = {metrics['roc_auc']:.3f}",
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve — WT vs KO classification")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    # Precision-Recall
    ax = axes[1]
    ax.plot(
        metrics["recall"], metrics["precision"],
        color="#534AB7", lw=2,
        label=f"AP = {metrics['average_precision']:.3f}",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve — WT vs KO classification")
    ax.legend(loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()
