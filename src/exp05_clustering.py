"""EXP-05: Unsupervised phenotype clustering of FIB-4 false negatives."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from sklearn.cluster import HDBSCAN as hdbscan

from src.utils.plot_utils import plot_umap_clusters, plot_radar_charts
from src.utils.nhanes_codebook import MODEL_FEATURES

# Raw, non-composite features used for FITTING (no double-counting).
# Composite scores (HSI, NFS, APRI, TYG, LBXLDL) are excluded here
# because they are linear functions of variables already in this list.
# They are still used for CHARACTERISATION (reporting medians, KW tests).
CLUSTERING_FEATURES = [
    # FIB-4 native (raw only)
    "RIDAGEYR", "LBXSASSI", "LBXSATSI", "LBXPLTSI",
    # Extended biochemistry (raw)
    "LBXSGTSI", "LBXSAL", "LBXSAPSI", "LBXSTB",
    "LBXSCR", "LBXSBU", "LBXSGL", "LBXSUA",
    # Metabolic (raw — LBXLDL excluded as correlated with LBXTC)
    "LBXGH", "LBXTR", "LBDHDD", "LBXTC",
    # Anthropometric (raw)
    "BMXBMI", "BMXWAIST", "WHTR",
    # Hematological (raw)
    "LBXHGB", "LBXWBCSI", "LBXMCVSI", "LBXRDW", "LBXMPSI",
    # Inflammatory (raw-ish, not simple functions of the above)
    "SII", "DE_RITIS",
]

# Diagnostic variables always added to the cluster profile output
# so LLM prompts receive FIB-4 score, LSM, sex, diabetes, race
EXTRA_PROFILE_VARS = ["FIB4", "LUXSMED", "RIAGENDR", "DIQ010", "RIDRETH3"]


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def gap_statistic(X: np.ndarray, k_range: list[int], n_refs: int = 20, random_state: int = 42) -> pd.DataFrame:
    """Compute gap statistic for K-Means cluster count selection."""
    rng = np.random.default_rng(random_state)
    gaps = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        wk = km.inertia_

        # Reference distribution
        ref_wks = []
        for _ in range(n_refs):
            ref = rng.uniform(X.min(axis=0), X.max(axis=0), size=X.shape)
            ref_km = KMeans(n_clusters=k, random_state=random_state, n_init=5)
            ref_km.fit(ref)
            ref_wks.append(ref_km.inertia_)

        gap = np.mean(np.log(np.array(ref_wks) + 1e-10)) - np.log(wk + 1e-10)
        gaps.append({"k": k, "gap": gap, "wk": wk})
    return pd.DataFrame(gaps)


def select_best_k(X_scaled: np.ndarray, k_range: list[int], random_state: int = 42) -> int:
    """Select k using silhouette score."""
    scores = {}
    for k in k_range:
        if k < 2 or k >= len(X_scaled):
            continue
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        scores[k] = silhouette_score(X_scaled, labels)
    if not scores:
        return 3
    best_k = max(scores, key=scores.__getitem__)
    print(f"Silhouette scores: {scores}")
    print(f"Selected k={best_k} (silhouette={scores[best_k]:.3f})")
    return best_k


def cluster_profile_stats(
    df: pd.DataFrame,
    features: list[str],
    cluster_col: str = "cluster",
) -> pd.DataFrame:
    """Median ± IQR for each feature by cluster."""
    rows = []
    for c in sorted(df[cluster_col].unique()):
        grp = df[df[cluster_col] == c]
        row = {"cluster": c, "n": len(grp)}
        for f in features:
            if f not in grp.columns:
                continue
            vals = grp[f].dropna()
            row[f"{f}_median"] = vals.median()
            row[f"{f}_q25"] = vals.quantile(0.25)
            row[f"{f}_q75"] = vals.quantile(0.75)
        rows.append(row)
    return pd.DataFrame(rows)


def kruskal_tests(
    df: pd.DataFrame,
    features: list[str],
    cluster_col: str = "cluster",
) -> pd.DataFrame:
    """Kruskal-Wallis test for each feature across clusters."""
    results = []
    clusters = sorted(df[cluster_col].unique())
    for f in features:
        if f not in df.columns:
            continue
        groups = [df[df[cluster_col] == c][f].dropna().values for c in clusters]
        groups = [g for g in groups if len(g) >= 5]
        if len(groups) < 2:
            continue
        h, p = stats.kruskal(*groups)
        results.append({"feature": f, "H_stat": h, "p_value": p})
    result_df = pd.DataFrame(results).sort_values("p_value")
    result_df["bonferroni_p"] = (result_df["p_value"] * len(result_df)).clip(upper=1.0)
    return result_df


def run_exp05(config: dict | None = None) -> pd.DataFrame:
    if config is None:
        config = load_config()

    data_dir = Path(config["paths"]["data_processed"])
    results_dir = Path(config["paths"]["results"]) / "exp05"
    results_dir.mkdir(parents=True, exist_ok=True)

    feature_df = pd.read_parquet(data_dir / "features_fib4_low_risk.parquet")

    # Target population: only false negatives
    fn_df = feature_df[feature_df["FIB4_FALSE_NEGATIVE"] == 1].copy()
    print(f"EXP-05: False negative population N={len(fn_df):,}")

    # Two separate feature lists: clean set for FITTING, full set for CHARACTERISATION
    feat_cols_cluster = [c for c in CLUSTERING_FEATURES if c in fn_df.columns]
    feat_cols_all = [c for c in MODEL_FEATURES if c in fn_df.columns]
    print(f"Clustering features (fitting): {len(feat_cols_cluster)}")
    print(f"Characterisation features: {len(feat_cols_all)}")

    X_raw = fn_df[feat_cols_cluster].fillna(fn_df[feat_cols_cluster].median())

    # Standardise for clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # --- Phase A: UMAP dimensionality reduction ---
    cfg_cl = config["clustering"]
    print("Running UMAP 2D...")
    umap_2d = UMAP(
        n_components=2,
        n_neighbors=cfg_cl["umap_n_neighbors"],
        min_dist=cfg_cl["umap_min_dist"],
        random_state=cfg_cl["umap_random_state"],
    )
    embedding_2d = umap_2d.fit_transform(X_scaled)

    print("Running UMAP 3D...")
    umap_3d = UMAP(
        n_components=3,
        n_neighbors=cfg_cl["umap_n_neighbors"],
        min_dist=cfg_cl["umap_min_dist"],
        random_state=cfg_cl["umap_random_state"],
    )
    embedding_3d = umap_3d.fit_transform(X_scaled)

    # --- Phase B: Cluster count selection ---
    k_range = cfg_cl["kmeans_k_range"]
    best_k = select_best_k(X_scaled, k_range, random_state=42)

    # HDBSCAN sweep
    hdbscan_results = []
    for min_size in cfg_cl["hdbscan_min_cluster_sizes"]:
        hdb = hdbscan(min_cluster_size=min_size)
        lbls = hdb.fit_predict(X_scaled)
        n_clusters = len(set(lbls)) - (1 if -1 in lbls else 0)
        noise_pct = (lbls == -1).mean() * 100
        if n_clusters >= 2:
            sil = silhouette_score(X_scaled[lbls != -1], lbls[lbls != -1]) if (lbls != -1).sum() > 10 else 0
        else:
            sil = 0
        hdbscan_results.append({
            "min_cluster_size": min_size, "n_clusters": n_clusters,
            "noise_pct": noise_pct, "silhouette": sil,
        })
    hdbscan_df = pd.DataFrame(hdbscan_results)
    hdbscan_df.to_csv(results_dir / "hdbscan_sweep.csv", index=False)
    print(f"HDBSCAN sweep:\n{hdbscan_df}")

    # Gap statistic
    gap_df = gap_statistic(X_scaled, k_range)
    gap_df.to_csv(results_dir / "gap_statistic.csv", index=False)

    # --- Phase C: Final clustering ---
    print(f"Final K-Means clustering with k={best_k}...")
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    cluster_labels = km_final.fit_predict(X_scaled)
    fn_df["cluster"] = cluster_labels

    # Also compute HDBSCAN with best min_cluster_size
    best_hdb_row = hdbscan_df.loc[hdbscan_df["silhouette"].idxmax()]
    best_min_size = int(best_hdb_row["min_cluster_size"])
    hdb_final = hdbscan(min_cluster_size=best_min_size)
    fn_df["cluster_hdbscan"] = hdb_final.fit_predict(X_scaled)

    # Save cluster assignments
    assign_df = fn_df[["SEQN", "cluster", "cluster_hdbscan"]].copy() if "SEQN" in fn_df.columns else fn_df[["cluster", "cluster_hdbscan"]].copy()
    assign_df.to_parquet(results_dir / "cluster_assignments.parquet", index=False)

    # --- Phase D: Cluster characterisation ---
    # Also merge in extra diagnostic variables (FIB4, LUXSMED, sex, DM, race)
    # from the full cohort so LLM prompts have real values
    cohort_path = Path(config["paths"]["data_processed"]) / "nhanes_masld_cohort.parquet"
    if cohort_path.exists():
        cohort = pd.read_parquet(cohort_path)
        extra_cols = [c for c in EXTRA_PROFILE_VARS if c in cohort.columns]
        if extra_cols and "SEQN" in fn_df.columns and "SEQN" in cohort.columns:
            fn_df = fn_df.merge(cohort[["SEQN"] + extra_cols], on="SEQN", how="left", suffixes=("", "_cohort"))
            # Use cohort values for any extra cols not already present
            for col in extra_cols:
                if col not in fn_df.columns or fn_df[col].isna().all():
                    fn_df[col] = fn_df.get(f"{col}_cohort", np.nan)
        print(f"Merged extra profile vars: {extra_cols}")

    # Profile on all features + extras
    all_profile_features = list(set(feat_cols_all + [c for c in EXTRA_PROFILE_VARS if c in fn_df.columns]))
    profiles = cluster_profile_stats(fn_df, all_profile_features, cluster_col="cluster")
    profiles.to_csv(results_dir / "cluster_profiles.csv", index=False)
    print(f"\nCluster sizes: {fn_df['cluster'].value_counts().sort_index().to_dict()}")

    kw_results = kruskal_tests(fn_df, feat_cols_all, cluster_col="cluster")
    kw_results.to_csv(results_dir / "cluster_statistical_tests.csv", index=False)
    sig_features = kw_results[kw_results["bonferroni_p"] < 0.05]["feature"].tolist()
    print(f"Significantly different features (Bonferroni p<0.05): {len(sig_features)}")

    # UMAP plot with cluster labels
    plot_umap_clusters(
        embedding_2d, cluster_labels,
        output_path=results_dir / "umap_clusters.png",
    )

    # Radar charts (use top-8 most significant clustering features for readability)
    radar_features = [f for f in sig_features if f in feat_cols_cluster][:8]
    if len(radar_features) < 3:
        radar_features = feat_cols_cluster[:8]
    median_profiles = profiles.set_index("cluster")[[f"{f}_median" for f in radar_features if f"{f}_median" in profiles.columns]]
    median_profiles.columns = [c.replace("_median", "") for c in median_profiles.columns]

    if len(median_profiles) > 0 and len(median_profiles.columns) >= 3:
        plot_radar_charts(
            median_profiles.reset_index(),
            features=list(median_profiles.columns),
            output_path=results_dir / "cluster_radar_charts.png",
        )

    print(f"\nEXP-05 complete → {results_dir}/")
    return fn_df


if __name__ == "__main__":
    cfg = load_config()
    run_exp05(cfg)
