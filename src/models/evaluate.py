"""
evaluate.py
-----------
Full evaluation suite for the ClauseGuard ML pipeline.
Run this in the Jupyter notebook (notebooks/evaluation.ipynb).

Usage:
    from src.models.evaluate import run_full_evaluation
    from src.utils.data_loader import load_data
    df = load_data()
    metrics = run_full_evaluation(df)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    mean_absolute_error, r2_score, mean_squared_error,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# ── Style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f0f1a",
    "axes.facecolor": "#1a1a2e",
    "axes.edgecolor": "#333355",
    "axes.labelcolor": "#e0e0ff",
    "xtick.color": "#a0a0cc",
    "ytick.color": "#a0a0cc",
    "text.color": "#e0e0ff",
    "grid.color": "#2a2a4a",
    "grid.alpha": 0.5,
    "font.family": "monospace",
})

ACCENT = "#6366f1"
DANGER = "#ef4444"
SUCCESS = "#22c55e"
WARN = "#f59e0b"


def run_full_evaluation(df: pd.DataFrame) -> dict:
    """
    Master evaluation runner. Call this from the notebook.
    Prints metrics and saves plots to outputs/.
    """
    import os
    os.makedirs("outputs", exist_ok=True)

    print("═" * 62)
    print("   CLAUSEGUARD — FULL ML EVALUATION REPORT")
    print("═" * 62)

    # Filter categories with enough samples for stratified split
    cat_counts = df["category"].value_counts()
    valid_cats = cat_counts[cat_counts >= 4].index
    df_eval = df[df["category"].isin(valid_cats)].copy().reset_index(drop=True)
    print(f"\n📦 Dataset: {len(df_eval)} clauses | {df_eval['category'].nunique()} categories")

    texts = df_eval["text"].tolist()
    categories = df_eval["category"].tolist()
    scores = df_eval["rating_score"].tolist()

    le = LabelEncoder()
    y = le.fit_transform(categories)

    # ── 1. CLASSIFIER ──────────────────────────────────────────────
    print("\n" + "─" * 62)
    print("  [1] CLAUSE CLASSIFIER  —  SVM + TF-IDF")
    print("─" * 62)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=8000, ngram_range=(1, 2),
            stop_words="english", sublinear_tf=True
        )),
        ("svm", SVC(kernel="linear", C=1.0, probability=True, class_weight="balanced")),
    ])

    # 5-fold stratified cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(pipeline, texts, y, cv=cv, scoring="f1_macro")
    cv_acc = cross_val_score(pipeline, texts, y, cv=cv, scoring="accuracy")

    print(f"\n  5-Fold CV  F1  (macro): {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")
    print(f"  5-Fold CV  Accuracy  : {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")

    # Train/test split for detailed report
    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=0.25, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    test_labels = sorted(set(y_test))
    print("\n  Per-Category Classification Report:\n")
    print(classification_report(
        y_test, y_pred,
        target_names=le.inverse_transform(test_labels),
        labels=test_labels,
    ))

    _plot_confusion_matrix(y_test, y_pred, le, test_labels)
    _plot_cv_scores(cv_f1, cv_acc)
    _plot_feature_importance(pipeline, le)

    # ── 2. RISK SCORER ─────────────────────────────────────────────
    print("─" * 62)
    print("  [2] RISK SCORER  —  Linear Regression on Human Ratings")
    print("─" * 62)

    tfidf = TfidfVectorizer(max_features=8000, stop_words="english")
    X_vec = tfidf.fit_transform(texts)
    X_tr, X_te, s_tr, s_te = train_test_split(X_vec, scores, test_size=0.25, random_state=42)
    reg = LinearRegression().fit(X_tr, s_tr)
    s_pred = reg.predict(X_te)

    mae = mean_absolute_error(s_te, s_pred)
    rmse = np.sqrt(mean_squared_error(s_te, s_pred))
    r2 = r2_score(s_te, s_pred)

    print(f"\n  Mean Absolute Error : {mae:.3f}  (on a 1–10 scale)")
    print(f"  Root Mean Sq. Error : {rmse:.3f}")
    print(f"  R² Score            : {r2:.3f}")

    _plot_risk_scatter(s_te, s_pred)
    _plot_risk_distribution(df_eval)

    # ── 3. DATA OVERVIEW ───────────────────────────────────────────
    print("─" * 62)
    print("  [3] DATASET OVERVIEW")
    print("─" * 62)
    _plot_data_overview(df_eval)

    metrics = {
        "cv_f1_mean": round(cv_f1.mean(), 4),
        "cv_f1_std": round(cv_f1.std(), 4),
        "cv_acc_mean": round(cv_acc.mean(), 4),
        "risk_mae": round(mae, 4),
        "risk_rmse": round(rmse, 4),
        "risk_r2": round(r2, 4),
        "n_clauses": len(df_eval),
        "n_categories": df_eval["category"].nunique(),
    }

    print("\n═" * 62)
    print("  ✅ Evaluation complete. Plots saved to outputs/")
    print("═" * 62)
    return metrics


# ── PRIVATE PLOT FUNCTIONS ─────────────────────────────────────────

def _plot_confusion_matrix(y_test, y_pred, le, test_labels):
    cm = confusion_matrix(y_test, y_pred, labels=test_labels)
    names = [n[:12] for n in le.inverse_transform(test_labels)]  # truncate long names

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d",
        cmap=sns.color_palette("Blues", as_cmap=True),
        xticklabels=names, yticklabels=names, ax=ax,
        linewidths=0.5, linecolor="#0f0f1a",
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Clause Classifier — Confusion Matrix", fontsize=14, fontweight="bold", pad=16)
    ax.set_ylabel("True Category", fontsize=11)
    ax.set_xlabel("Predicted Category", fontsize=11)
    plt.xticks(rotation=40, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  → Saved: outputs/confusion_matrix.png")


def _plot_cv_scores(cv_f1, cv_acc):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, scores, title, color in zip(
        axes,
        [cv_f1, cv_acc],
        ["5-Fold F1 Score (macro)", "5-Fold Accuracy"],
        [ACCENT, SUCCESS],
    ):
        folds = [f"Fold {i+1}" for i in range(len(scores))]
        bars = ax.bar(folds, scores, color=color, alpha=0.85, edgecolor="#0f0f1a")
        ax.axhline(scores.mean(), color=WARN, ls="--", lw=1.5, label=f"Mean: {scores.mean():.3f}")
        ax.set_ylim(0, 1.1)
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        for bar, val in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9, color="white")

    plt.suptitle("Cross-Validation Results — SVM Classifier", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/cv_scores.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  → Saved: outputs/cv_scores.png")


def _plot_feature_importance(pipeline, le):
    tfidf = pipeline.named_steps["tfidf"]
    svm = pipeline.named_steps["svm"]
    feature_names = tfidf.get_feature_names_out()

    n_classes = len(svm.classes_)
    if n_classes < 2:
        return

    # Use first vs rest coefficients
    coefs = svm.coef_
    mean_coef = np.abs(coefs).mean(axis=0)
    top_n = 20
    top_idx = np.argsort(mean_coef)[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [DANGER if mean_coef[i] > mean_coef.mean() else ACCENT for i in top_idx]
    ax.barh(
        [feature_names[i] for i in top_idx],
        mean_coef[top_idx],
        color=colors, edgecolor="#0f0f1a", alpha=0.9,
    )
    ax.set_title(f"Top {top_n} Most Predictive Terms (TF-IDF Weights)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean |Coefficient| across classes")

    high_patch = mpatches.Patch(color=DANGER, label="High discriminative power")
    low_patch = mpatches.Patch(color=ACCENT, label="Moderate discriminative power")
    ax.legend(handles=[high_patch, low_patch], fontsize=9)

    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  → Saved: outputs/feature_importance.png")


def _plot_risk_scatter(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(
        y_true, y_pred,
        c=y_pred, cmap="RdYlGn_r",
        alpha=0.7, edgecolors="#0f0f1a", s=60,
    )
    ax.plot([1, 10], [1, 10], color=WARN, ls="--", lw=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Risk Score (Human Label)", fontsize=11)
    ax.set_ylabel("Predicted Risk Score (Model)", fontsize=11)
    ax.set_title("Risk Scorer — Predicted vs Actual", fontsize=13, fontweight="bold")
    ax.legend()
    plt.colorbar(scatter, ax=ax, label="Predicted Risk")
    plt.tight_layout()
    plt.savefig("outputs/risk_scatter.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  → Saved: outputs/risk_scatter.png")


def _plot_risk_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Rating distribution
    rating_counts = df["rating"].value_counts()
    colors_map = {"good": SUCCESS, "neutral": ACCENT, "bad": WARN, "very bad": DANGER, "blocker": "#7f1d1d"}
    bar_colors = [colors_map.get(r.lower(), ACCENT) for r in rating_counts.index]
    axes[0].bar(rating_counts.index, rating_counts.values, color=bar_colors, edgecolor="#0f0f1a", alpha=0.9)
    axes[0].set_title("Clause Rating Distribution", fontweight="bold")
    axes[0].set_ylabel("Count")

    # Category distribution
    cat_counts = df["category"].value_counts().head(8)
    axes[1].barh(cat_counts.index, cat_counts.values, color=ACCENT, edgecolor="#0f0f1a", alpha=0.9)
    axes[1].set_title("Top 8 Clause Categories", fontweight="bold")
    axes[1].set_xlabel("Count")

    plt.suptitle("Training Data Overview", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/data_overview.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  → Saved: outputs/data_overview.png")


def _plot_data_overview(df):
    _plot_risk_distribution(df)
