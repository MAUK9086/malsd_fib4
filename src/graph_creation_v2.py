"""Graph creation v2 — 4-panel main figure for publication.

Panels:
  A: SHAP beeswarm (raw features, EXP-11)
  B: BMI-stratified FN rates bar chart (EXP-12)
  C: FIB-4 Blindspot Score forest plot (EXP-13)
  D: LLM-SHAP concordance heatmap (EXP-15 preferred, fallback EXP-07)

Fixes axs[1, 0].axis('off') bug from v1 (was axs[1, 1]).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import yaml


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_panel_image(path: Path, ax: plt.Axes, title: str = "") -> bool:
    """Load a pre-rendered PNG into an axes. Returns True on success."""
    if path.exists():
        img = mpimg.imread(str(path))
        ax.imshow(img, aspect="auto")
        ax.axis("off")
        if title:
            ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
        return True
    ax.axis("off")
    ax.text(0.5, 0.5, f"[Panel not available]\n{path.name}",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=8, color="grey")
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    return False


def render_bmi_bar_chart(results_dir_12: Path, ax: plt.Axes) -> bool:
    """Render BMI-stratified FN rate bar chart directly if PNG not available."""
    csv_path = results_dir_12 / "bmi_group_fn_rates.csv"
    if not csv_path.exists():
        return False
    df = pd.read_csv(csv_path)
    bars = ax.bar(df["bmi_group"].astype(str), df["fn_rate_pct"],
                  color=["#4E79A7", "#F28E2B", "#E15759", "#76B7B2"], alpha=0.85, edgecolor="white")
    ax.set_xlabel("BMI group (kg/m²)", fontsize=9)
    ax.set_ylabel("FIB-4 False Negative Rate (%)", fontsize=9)
    ax.set_title("B. FN Rate by BMI Group", fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, df["fn_rate_pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
    return True


def render_blindspot_table(results_dir_13: Path, ax: plt.Axes) -> bool:
    """Render Blindspot Score criteria as a formatted table."""
    criteria = [
        ("BMI ≥ 30 kg/m²", "+1"),
        ("HbA1c ≥ 5.7%\n(prediabetes)", "+1"),
        ("Waist ≥ 102/88 cm\n(men/women)", "+1"),
        ("GGT > 36/25 U/L\n(men/women)", "+1"),
    ]
    # Load AUROC if available
    auroc_text = ""
    bs_path = results_dir_13 / "blindspot_score_auroc.csv"
    if bs_path.exists():
        try:
            bs_df = pd.read_csv(bs_path)
            bs_row = bs_df[bs_df["model"].str.contains("Blindspot", case=False)]
            if len(bs_row) > 0:
                auroc_text = f"AUROC = {bs_row['auroc'].iloc[0]:.3f}"
        except Exception:
            pass

    png_path = results_dir_13 / "fib4_blindspot_figure.png"
    if png_path.exists():
        img = mpimg.imread(str(png_path))
        ax.imshow(img, aspect="auto")
        ax.axis("off")
        ax.set_title("C. FIB-4 Blindspot Predictors", fontsize=10, fontweight="bold", pad=4)
        return True

    # Fallback: draw table
    ax.axis("off")
    ax.set_title("C. FIB-4 Blindspot Score (0–4)", fontsize=10, fontweight="bold")
    y_start = 0.85
    dy = 0.18
    for i, (criterion, score) in enumerate(criteria):
        y = y_start - i * dy
        ax.text(0.05, y, criterion, transform=ax.transAxes, fontsize=8, va="top")
        ax.add_patch(plt.Rectangle(
            (0.78, y - 0.12), 0.14, 0.14,
            transform=ax.transAxes, fc="#4E79A7", ec="white", alpha=0.8
        ))
        ax.text(0.85, y - 0.05, score, transform=ax.transAxes,
                fontsize=11, fontweight="bold", ha="center", va="top", color="white")
    if auroc_text:
        ax.text(0.5, 0.02, auroc_text, transform=ax.transAxes,
                fontsize=9, ha="center", va="bottom", color="#333333",
                style="italic")
    return True


def create_main_figure_v2(config: dict | None = None) -> None:
    if config is None:
        config = load_config()

    results_root = Path(config["paths"]["results"])
    fig_dir = results_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    results_dir_11 = results_root / "exp11"
    results_dir_12 = results_root / "exp12"
    results_dir_13 = results_root / "exp13"
    results_dir_15 = results_root / "exp15"
    results_dir_07 = results_root / "exp07"

    fig = plt.figure(figsize=(14, 11))
    fig.suptitle(
        "FIB-4 Phenotyping Study — Main Figure",
        fontsize=13, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.25,
                           left=0.06, right=0.97, top=0.94, bottom=0.04)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # --- Panel A: SHAP beeswarm (raw features, EXP-11) ---
    shap_path = results_dir_11 / "shap_beeswarm_raw.png"
    ok_a = load_panel_image(shap_path, ax_a, "A. Top Raw SHAP Features (EXP-11)")
    if not ok_a:
        # Fallback: EXP-04 SHAP bar
        alt = results_root / "exp04" / "shap_beeswarm.png"
        load_panel_image(alt, ax_a, "A. SHAP Feature Importance (EXP-04)")

    # --- Panel B: BMI-stratified FN rates (EXP-12) ---
    bmi_png = results_dir_12 / "bmi_group_fn_rates.png"
    if bmi_png.exists():
        load_panel_image(bmi_png, ax_b, "B. FN Rate by BMI Group (EXP-12)")
    else:
        rendered = render_bmi_bar_chart(results_dir_12, ax_b)
        if not rendered:
            ax_b.axis("off")
            ax_b.text(0.5, 0.5, "[Run EXP-12 to generate]",
                      ha="center", va="center", transform=ax_b.transAxes,
                      fontsize=9, color="grey")
            ax_b.set_title("B. FN Rate by BMI Group", fontsize=10, fontweight="bold")

    # --- Panel C: Blindspot Score (EXP-13) ---
    render_blindspot_table(results_dir_13, ax_c)

    # --- Panel D: LLM–SHAP concordance heatmap ---
    # EXP-15 first, fallback EXP-07
    concordance_loaded = False
    for conc_dir, label in [(results_dir_15, "EXP-15 v2"), (results_dir_07, "EXP-07")]:
        heatmap_path = conc_dir / "concordance_heatmap.png"
        if heatmap_path.exists():
            load_panel_image(heatmap_path, ax_d,
                             f"D. LLM–SHAP Concordance Heatmap ({label})")
            concordance_loaded = True
            break
    if not concordance_loaded:
        # Draw from CSV if PNG not available
        for conc_csv, label in [
            (results_dir_15 / "llm_shap_concordance_v2.csv", "v2"),
            (results_dir_07 / "llm_shap_concordance.csv", "v1"),
        ]:
            if conc_csv.exists():
                conc_df = pd.read_csv(conc_csv)
                if "cluster_id" in conc_df.columns and "model" in conc_df.columns:
                    pivot = (
                        conc_df[conc_df["prompt_type"] == "PHENOTYPE"]
                        .groupby(["cluster_id", "model"])["concordance"]
                        .mean()
                        .unstack("model")
                    )
                    if len(pivot) > 0:
                        im = ax_d.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                                         vmin=0, vmax=1)
                        ax_d.set_xticks(range(len(pivot.columns)))
                        ax_d.set_xticklabels(pivot.columns, fontsize=7, rotation=30)
                        ax_d.set_yticks(range(len(pivot.index)))
                        ax_d.set_yticklabels([f"Cluster {i}" for i in pivot.index], fontsize=8)
                        plt.colorbar(im, ax=ax_d, shrink=0.8, label="Concordance")
                        ax_d.set_title(f"D. LLM–SHAP Concordance ({label})",
                                       fontsize=10, fontweight="bold")
                        concordance_loaded = True
                        break
        if not concordance_loaded:
            ax_d.axis("off")
            ax_d.text(0.5, 0.5, "[Run EXP-15 to generate]",
                      ha="center", va="center", transform=ax_d.transAxes,
                      fontsize=9, color="grey")
            ax_d.set_title("D. LLM–SHAP Concordance", fontsize=10, fontweight="bold")

    out_path = fig_dir / "main_figure_v2.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved main figure v2 → {out_path}")


if __name__ == "__main__":
    cfg = load_config()
    create_main_figure_v2(cfg)
