"""EXP-01: Merge all NHANES XPT files on SEQN → nhanes_merged_raw.parquet."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyreadstat
import yaml


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def read_xpt(path: Path) -> pd.DataFrame:
    """Read a SAS XPT file, return DataFrame with SEQN as int index."""
    try:
        df, _ = pyreadstat.read_xport(str(path))
    except Exception:
        # Fallback to pandas
        df = pd.read_sas(str(path), format="xport", encoding="utf-8")
    if "SEQN" in df.columns:
        df["SEQN"] = df["SEQN"].astype(int)
    return df


def merge_nhanes(config: dict) -> pd.DataFrame:
    raw_dir = Path(config["paths"]["data_raw"])
    data_dir = Path(config["paths"]["data_processed"])
    data_dir.mkdir(parents=True, exist_ok=True)

    # Anchor on P_LUX — only participants with FibroScan
    lux_path = raw_dir / "P_LUX.XPT"
    if not lux_path.exists():
        # Try fallback name
        lux_path = raw_dir / "LUX_J.XPT"
    if not lux_path.exists():
        raise FileNotFoundError(
            "P_LUX.XPT not found. Run download_nhanes.py first."
        )

    print(f"Anchor: {lux_path}")
    df = read_xpt(lux_path)
    print(f"  P_LUX: {len(df):,} rows")

    # All other files to merge in
    merge_files = [
        "P_BIOPRO", "P_CBC", "P_GHB", "P_DEMO", "P_BMX",
        "P_DIQ", "P_ALQ", "P_HEQ", "P_TRIGLY", "P_HDL",
        "P_TCHOL", "P_BPX", "P_SMQ",
    ]

    for prefix in merge_files:
        candidate_paths = [
            raw_dir / f"{prefix}.XPT",
            raw_dir / f"{prefix.replace('P_', '')}_J.XPT",
        ]
        found_path = next((p for p in candidate_paths if p.exists()), None)
        if found_path is None:
            print(f"  WARNING: {prefix} not found — skipping")
            continue

        right = read_xpt(found_path)
        # Drop duplicate columns except SEQN
        overlap = [c for c in right.columns if c in df.columns and c != "SEQN"]
        if overlap:
            right = right.drop(columns=overlap)

        before = len(df)
        df = df.merge(right, on="SEQN", how="left")
        print(f"  Merged {prefix}: {len(right):,} rows → {len(df):,} kept")

    out_path = data_dir / "nhanes_merged_raw.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved merged dataset: {out_path}  ({len(df):,} rows × {len(df.columns)} cols)")
    return df


if __name__ == "__main__":
    cfg = load_config()
    df = merge_nhanes(cfg)
    print(df.shape)
    print(df.head(2))
