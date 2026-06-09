"""EXP-01: Download all required NHANES 2017-2020 XPT files."""

from __future__ import annotations

import time
from pathlib import Path

import requests
import yaml
from tqdm import tqdm


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def download_file(url: str, dest: Path, timeout: int = 60) -> bool:
    """Download a single file with progress bar. Returns True on success."""
    try:
        resp = requests.get(url, stream=True, timeout=timeout)
        if resp.status_code != 200:
            return False
        total = int(resp.headers.get("content-length", 0))
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True,
            desc=dest.name, leave=False
        ) as bar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        return True
    except Exception as exc:
        print(f"  Error: {exc}")
        return False


def download_with_retry(
    url: str,
    dest: Path,
    max_retries: int = 4,
) -> bool:
    """Download with exponential backoff retry."""
    for attempt in range(max_retries + 1):
        if download_file(url, dest):
            return True
        if attempt < max_retries:
            wait = 2 ** attempt
            print(f"  Retry {attempt+1}/{max_retries} in {wait}s...")
            time.sleep(wait)
    return False


def download_nhanes_files(config: dict) -> dict[str, Path]:
    """
    Download all NHANES XPT files.
    For each file, try the preferred P_* name first, then fall back to _J suffix.
    Returns dict of {file_prefix: local_path}.
    """
    base_url = config["nhanes"]["base_url"].rstrip("/")
    raw_dir = Path(config["paths"]["data_raw"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    downloaded: dict[str, Path] = {}
    failed: list[str] = []

    file_list = config["nhanes"]["files"]
    print(f"Downloading {len(file_list)} NHANES files to {raw_dir}/")

    for entry in file_list:
        preferred, fallback = entry[0], entry[1]
        dest = raw_dir / f"{preferred}.XPT"

        # Skip if already downloaded
        if dest.exists():
            print(f"  [SKIP] {preferred}.XPT already exists")
            downloaded[preferred] = dest
            continue

        # Try preferred URL (P_ prefix)
        url_primary = f"{base_url}/{preferred}.XPT"
        print(f"  Trying {url_primary}")
        ok = download_with_retry(url_primary, dest)

        if not ok:
            # Try 2019-2020 cycle URL
            url_alt_2020 = f"https://wwwn.cdc.gov/nchs/nhanes/2019/DataFiles/{preferred}.XPT"
            print(f"  Trying alternate 2019 URL: {url_alt_2020}")
            ok = download_with_retry(url_alt_2020, dest)

        if not ok:
            # Fall back to 2017-2018 suffix (_J)
            fallback_dest = raw_dir / f"{fallback}.XPT"
            url_fallback = f"https://wwwn.cdc.gov/nchs/nhanes/2017/DataFiles/{fallback}.XPT"
            print(f"  Falling back to {url_fallback}")
            ok = download_with_retry(url_fallback, fallback_dest)
            if ok:
                print(f"  [OK-FALLBACK] {fallback}.XPT")
                downloaded[preferred] = fallback_dest
            else:
                print(f"  [FAILED] {preferred} and {fallback}")
                failed.append(preferred)
        else:
            print(f"  [OK] {preferred}.XPT")
            downloaded[preferred] = dest

    print(f"\nDownload summary: {len(downloaded)}/{len(file_list)} files obtained")
    if failed:
        print(f"FAILED files (manual download required): {failed}")

    # Write download log
    log_dir = Path(config["paths"]["results"]) / "exp01"
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "download_log.txt", "w") as f:
        f.write("NHANES Download Log\n")
        f.write("=" * 40 + "\n")
        for prefix, path in downloaded.items():
            f.write(f"OK: {prefix} → {path}\n")
        for prefix in failed:
            f.write(f"FAILED: {prefix}\n")

    return downloaded


if __name__ == "__main__":
    cfg = load_config()
    downloaded = download_nhanes_files(cfg)
    print(f"\nReady: {len(downloaded)} files in {cfg['paths']['data_raw']}/")
