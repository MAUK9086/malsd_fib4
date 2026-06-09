"""Visualisation helpers — all plots save to results/expXX/ paths."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Style defaults
# ---------------------------------------------------------------------------

PALETTE = sns.color_palette("tab10")

def _save(fig: plt.Figure, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# ROC curve
# ---------------------------------------------------------------------------

def plot_roc_curves(
    curves: list[dict],  # [{"label": str, "fpr": list, "tpr": list, "auroc": float}]
    output_path: str | Path,
    title: str = "ROC Curves",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, c in enumerate(curves):
        ax.plot(
            c["fpr"], c["tpr"],
            label=f'{c["label"]} (AUC={c["auroc"]:.3f})',
            color=PALETTE[i],
        )
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")
    ax.set_xlabel("1 − Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Calibration curve
# ---------------------------------------------------------------------------

def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: str | Path,
    n_bins: int = 10,
    title: str = "Calibration Curve",
) -> None:
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, "s-", label="Model", color=PALETTE[0])
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction Positive")
    ax.set_title(title)
    ax.legend()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# SHAP beeswarm (wrapper around shap's native plot)
# ---------------------------------------------------------------------------

def plot_shap_summary(
    shap_values,
    feature_data: pd.DataFrame,
    output_path: str | Path,
    title: str = "SHAP Summary",
    max_display: int = 20,
) -> None:
    import shap
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        feature_data,
        show=False,
        max_display=max_display,
        plot_type="dot",
    )
    plt.title(title)
    _save(fig, output_path)


def plot_shap_bar(
    shap_values,
    feature_names: list[str],
    output_path: str | Path,
    title: str = "SHAP Feature Importance",
    max_display: int = 20,
) -> None:
    import shap
    fig = plt.figure(figsize=(8, 7))
    shap.summary_plot(
        shap_values,
        feature_names=feature_names,
        show=False,
        plot_type="bar",
        max_display=max_display,
    )
    plt.title(title)
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# UMAP scatter
# ---------------------------------------------------------------------------

def plot_umap_clusters(
    embedding: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path,
    title: str = "UMAP — FIB-4 False Negatives",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    unique_labels = np.unique(labels)
    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        label_str = f"Noise" if lbl == -1 else f"Cluster {lbl}"
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=20,
            alpha=0.7,
            color=PALETTE[i % len(PALETTE)],
            label=label_str,
        )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(title)
    ax.legend(markerscale=2, fontsize=9)
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Radar chart for cluster profiles
# ---------------------------------------------------------------------------

def plot_radar_charts(
    cluster_profiles: pd.DataFrame,  # rows=clusters, cols=features
    features: list[str],
    output_path: str | Path,
    feature_labels: list[str] | None = None,
) -> None:
    if feature_labels is None:
        feature_labels = features
    n_features = len(features)
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(
        1, len(cluster_profiles),
        figsize=(5 * len(cluster_profiles), 5),
        subplot_kw={"polar": True},
    )
    if len(cluster_profiles) == 1:
        axes = [axes]

    # Normalise to [0, 1] across clusters for each feature
    norm_profiles = cluster_profiles[features].copy()
    for col in features:
        col_min, col_max = norm_profiles[col].min(), norm_profiles[col].max()
        if col_max > col_min:
            norm_profiles[col] = (norm_profiles[col] - col_min) / (col_max - col_min)
        else:
            norm_profiles[col] = 0.5

    for idx, (ax, (_, row)) in enumerate(zip(axes, norm_profiles.iterrows())):
        values = row[features].tolist() + row[features].tolist()[:1]
        ax.plot(angles, values, "o-", color=PALETTE[idx % len(PALETTE)])
        ax.fill(angles, values, alpha=0.25, color=PALETTE[idx % len(PALETTE)])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_labels, size=7)
        ax.set_ylim(0, 1)
        ax.set_title(f"Cluster {idx}", size=10, pad=15)

    plt.suptitle("Cluster Phenotype Radar Charts", y=1.02)
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    df: pd.DataFrame,
    output_path: str | Path,
    title: str = "Feature Correlation Heatmap",
) -> None:
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr,
        ax=ax,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.2,
        annot=False,
    )
    ax.set_title(title)
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(fontsize=7)
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Feature distribution grid
# ---------------------------------------------------------------------------

def plot_feature_distributions(
    df: pd.DataFrame,
    features: list[str],
    hue_col: str | None,
    output_path: str | Path,
    ncols: int = 5,
) -> None:
    nrows = (len(features) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.5))
    axes_flat = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes.flatten()

    for ax, feat in zip(axes_flat, features):
        if hue_col and hue_col in df.columns:
            for val, grp in df.groupby(hue_col):
                ax.hist(grp[feat].dropna(), bins=30, alpha=0.5, label=str(val), density=True)
            ax.legend(fontsize=6)
        else:
            ax.hist(df[feat].dropna(), bins=30, color=PALETTE[0], alpha=0.7)
        ax.set_title(feat, fontsize=8)
        ax.set_xlabel("")
    for ax in axes_flat[len(features):]:
        ax.set_visible(False)

    plt.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Decision curve analysis
# ---------------------------------------------------------------------------

def plot_decision_curve(
    y_true: np.ndarray,
    score_dict: dict[str, np.ndarray],
    output_path: str | Path,
    title: str = "Decision Curve Analysis",
) -> None:
    thresholds = np.linspace(0.01, 0.99, 100)
    n = len(y_true)
    prev = y_true.mean()

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (name, scores) in enumerate(score_dict.items()):
        net_benefits = []
        for t in thresholds:
            tp = ((scores >= t) & (y_true == 1)).sum()
            fp = ((scores >= t) & (y_true == 0)).sum()
            nb = tp / n - fp / n * (t / (1 - t)) if t < 1 else 0
            net_benefits.append(nb)
        ax.plot(thresholds, net_benefits, label=name, color=PALETTE[i])

    # Treat all as positive baseline
    treat_all = prev - (1 - prev) * thresholds / (1 - thresholds + 1e-10)
    ax.plot(thresholds, treat_all, "k--", label="Treat All")
    ax.axhline(0, color="gray", linewidth=0.8, label="Treat None")

    ax.set_xlabel("Threshold Probability")
    ax.set_ylabel("Net Benefit")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Concordance heatmap (EXP-07)
# ---------------------------------------------------------------------------

def plot_concordance_heatmap(
    concordance_df: pd.DataFrame,  # rows=clusters, cols=model, values=concordance
    output_path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(max(5, len(concordance_df.columns) * 2), max(4, len(concordance_df) * 1.2)))
    sns.heatmap(
        concordance_df,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="YlGn",
        vmin=0,
        vmax=1,
        linewidths=0.5,
    )
    ax.set_title("LLM–SHAP Concordance by Cluster and Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Cluster")
    _save(fig, output_path)
