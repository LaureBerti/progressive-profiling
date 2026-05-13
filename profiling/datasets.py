"""Dataset loaders for D1–D5, D7 (synthetic, real admin, scalability, IoT)."""
from __future__ import annotations

import json
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple

from .data_generator import (
    SyntheticTabularGenerator, IoTSensorGenerator, ErrorConfig,
    generate_sdv_dataset, inject_errors_into_df,
)
from .quality_profile import QualityProfile, compute_quality_profile

# Base directory for Outsourcing pre-generated synthetic files
# Pattern: synthetic_N{n}_seed{s}.parquet
_OUTSOURCING_BASE = Path(__file__).parent.parent.parent / "Outsourcing" / "outputs" / "E4" / "synthetic"


def _cache_path(data_dir: Path, name: str) -> Path:
    return data_dir / f"{name}.parquet"


def load_dataset(
    name: str,
    data_dir: str = "data",
    n_records: Optional[int] = None,
    seed: int = 42,
    error_rate: float = 0.05,
    correlation_rho: float = 0.0,
    force_regen: bool = False,
) -> Tuple[pd.DataFrame, Optional[QualityProfile], Dict[str, str]]:
    """Load a named dataset.

    Returns (df, ground_truth_profile, fd_rules).
    ground_truth_profile is None for real datasets (must be estimated via exhaustive profiling).
    """
    data_path = Path(data_dir)
    loaders = {
        "D1": _load_d1,   # synthetic controlled
        "D2": _load_d2,   # NYC 311 service requests
        "D3": _load_d3,   # NYPD Arrest Data
        "D4": _load_d4,   # UCI Adult census income
        "D5": _load_d5,   # Adult-derived synthetic scalability
        # D7 variants — IoT sensor data
        "D7":       _load_d7,        # auto: real if cache present, else synthetic
        "D7-synth": _load_d7_synth,  # force synthetic
        "D7-real":  _load_d7_real,   # force real (Intel Lab or user-supplied)
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name!r}. Known: {list(loaders)}")
    return loaders[name](data_path, n_records, seed, error_rate, correlation_rho, force_regen)


def _load_d1(data_path, n_records, seed, error_rate, rho, force_regen):
    """D1: numpy-based synthetic tabular with controlled error injection."""
    n = n_records or 100_000
    rng = np.random.default_rng(seed)
    gen = SyntheticTabularGenerator(rng=rng)
    cfg = ErrorConfig(
        missing_rate=error_rate,
        duplicate_rate=error_rate * 0.5,
        outlier_rate=error_rate * 0.5,
        inconsistency_rate=error_rate * 0.3,
        correlation_rho=rho,
    )
    dirty, clean, fd_rules = gen.generate(n, n_cols=8, config=cfg)

    # Ground truth: exhaustive profile on clean data with known error rates
    gt = QualityProfile(
        missing_rate=cfg.missing_rate,
        duplicate_rate=cfg.duplicate_rate,
        outlier_rate=cfg.outlier_rate,
        inconsistency_rate=cfg.inconsistency_rate,
        n_records=n,
        sample_size=n,
    )
    return dirty, gt, fd_rules


def _load_d_wiki(data_path, n_records, seed, error_rate, rho, force_regen):
    """D_wiki: Wikidata RDF triples, converted to entity–property–value tabular form.

    Expects pre-processed Parquet at data/D_wiki_wikidata.parquet.
    If not present, uses a small stub with a warning.
    """
    cache = data_path / "D_wiki_wikidata.parquet"
    return _load_real_or_stub(cache, "D_wiki_wikidata", n_records, seed, n_stub=50_000)


def _load_d_dbp(data_path, n_records, seed, error_rate, rho, force_regen):
    """D_dbp: DBpedia RDF subset, entity–property–value tabular form."""
    cache = data_path / "D_dbp_dbpedia.parquet"
    return _load_real_or_stub(cache, "D_dbp_dbpedia", n_records, seed, n_stub=50_000)


def _load_d2(data_path, n_records, seed, error_rate, rho, force_regen):
    """D2: NYC 311 service requests (tabular, ~10M rows)."""
    cache = data_path / "D2_nyc311.parquet"
    return _load_real_or_stub(cache, "D2_nyc311", n_records, seed, n_stub=50_000)


def _load_d3(data_path, n_records, seed, error_rate, rho, force_regen):
    """D3: NCDC weather data with known quality issues."""
    cache = data_path / "D3_ncdc.parquet"
    return _load_real_or_stub(cache, "D3_ncdc", n_records, seed, n_stub=50_000)


def _load_d5(data_path, n_records, seed, error_rate, rho, force_regen):
    """D5: Pre-generated Adult-derived tabular data for scaling experiments (E5).

    Loads from Outsourcing/outputs/E4/synthetic/synthetic_N{n}_seed{s}.parquet when
    available (n in {100K, 500K, 1M, 5M}). For n=10K, subsamples from n=100K file.
    Injects controlled errors and caches dirty data + GT profile locally.
    Falls back to SDV generation if Outsourcing files are not found.
    """
    n = n_records or 100_000
    cache = data_path / f"D5_outsourcing_n{n}_s{seed}.parquet"
    gt_path = cache.with_suffix(".gt.json")

    if cache.exists() and not force_regen:
        df = pd.read_parquet(cache)
        gt = _load_gt_json(gt_path, len(df))
        return df, gt, {}  # GT was computed with fd_rules={}; keep consistent

    # Resolve source file from Outsourcing pre-generated files
    outsourcing_file = _outsourcing_file_for(n, seed)
    if outsourcing_file is not None:
        clean = pd.read_parquet(outsourcing_file)
        if n < len(clean):
            clean = clean.sample(n=n, random_state=seed).reset_index(drop=True)
    else:
        # Fallback: SDV or numpy generation
        import warnings
        warnings.warn(
            f"D5 Outsourcing file not found for n={n}, seed={seed}. Falling back to SDV/numpy.",
            stacklevel=3,
        )
        dirty, clean_df, fd_rules = generate_sdv_dataset(n, seed=seed, error_rate=error_rate)
        cache.parent.mkdir(parents=True, exist_ok=True)
        dirty.to_parquet(cache, index=False)
        gt = compute_quality_profile(dirty, fd_rules)
        gt.n_records = len(dirty)
        _save_gt_json(gt_path, gt)
        return dirty, gt, fd_rules

    rng = np.random.default_rng(seed + 7)
    dirty, fd_rules = inject_errors_into_df(clean, rng, error_rate=error_rate)
    cache.parent.mkdir(parents=True, exist_ok=True)
    dirty.to_parquet(cache, index=False)
    gt = compute_quality_profile(dirty, fd_rules)
    gt.n_records = len(dirty)
    _save_gt_json(gt_path, gt)
    return dirty, gt, fd_rules


def _outsourcing_file_for(n: int, seed: int) -> Optional[Path]:
    """Return path to Outsourcing pre-generated file, or None if not available."""
    # For n ≤ 10K, we'll subsample from n=100K
    source_n = n if n >= 100_000 else 100_000
    candidate = _OUTSOURCING_BASE / f"synthetic_N{source_n}_seed{seed}.parquet"
    if candidate.exists():
        return candidate
    # Try other seeds that are available (for n=5M only seed=42 exists)
    alt = _OUTSOURCING_BASE / f"synthetic_N{source_n}_seed42.parquet"
    if source_n >= 5_000_000 and alt.exists():
        return alt
    return None


def _load_real_or_stub(
    cache: Path,
    name: str,
    n_records: Optional[int],
    seed: int,
    n_stub: int = 500_000,
) -> Tuple[pd.DataFrame, Optional[QualityProfile], Dict[str, str]]:
    if cache.exists():
        df = pd.read_parquet(cache)
        if n_records and len(df) > n_records:
            df = df.sample(n=n_records, random_state=seed).reset_index(drop=True)
        return df, None, {}

    import warnings
    warnings.warn(
        f"{name}: Parquet cache not found at {cache}. "
        f"Generating a stub dataset. See README for download instructions.",
        stacklevel=3,
    )
    n = n_records or n_stub
    rng = np.random.default_rng(seed)
    gen = SyntheticTabularGenerator(rng=rng)
    cfg = ErrorConfig(missing_rate=0.04, duplicate_rate=0.01, outlier_rate=0.02, inconsistency_rate=0.01)
    dirty, _, fd_rules = gen.generate(n, n_cols=10, config=cfg)
    return dirty, None, fd_rules


def _load_d4(data_path, n_records, seed, error_rate, rho, force_regen):
    """D4: UCI Adult Income tabular data (non-NYC, 48K rows, mixed types).

    Downloaded from UCI ML Repository; missing values encoded as ' ?' → NaN.
    Download script: outputs/paper_ready/scripts/download_d4_adult.py
    """
    cache = data_path / "D4_adult.parquet"
    return _load_real_or_stub(cache, "D4_adult", n_records, seed, n_stub=48_842)


def _load_d7(data_path, n_records, seed, error_rate, rho, force_regen):
    """D7: IoT sensor data — use real cache if present, else synthetic."""
    real_cache = data_path / "D7_iot_real.parquet"
    if real_cache.exists() and not force_regen:
        return _load_d7_real(data_path, n_records, seed, error_rate, rho, force_regen)
    return _load_d7_synth(data_path, n_records, seed, error_rate, rho, force_regen)


def _load_d7_synth(data_path, n_records, seed, error_rate, rho, force_regen):
    """D7-synth: Synthetic IoT sensor dataset.

    11 columns: 8 numeric (temperature_c, humidity_pct, pressure_hpa, co2_ppm,
    light_lux, voltage_v, current_ma, battery_pct) + 3 categorical (device_id,
    location_zone, alert_level).

    Quality defects are numeric-dominant (sensor malfunctions → IQR-visible outliers),
    contrasting with D2/D3/D4 where defects concentrate in categorical columns.
    FD rule: device_id → location_zone.

    Default size: 500K rows (matches D2/D3 for direct comparison).
    Scalable to 10M+ rows for E5-style experiments.
    """
    n = n_records or 500_000
    cache = data_path / f"D7_synth_n{n}_s{seed}.parquet"
    gt_path = cache.with_suffix(".gt.json")

    if cache.exists() and not force_regen:
        df = pd.read_parquet(cache)
        gt = _load_gt_json(gt_path, len(df))
        return df, gt, {"device_id": "location_zone"}

    rng = np.random.default_rng(seed)
    gen = IoTSensorGenerator(rng=rng)
    cfg = ErrorConfig(
        missing_rate=max(error_rate, 0.08),
        duplicate_rate=max(error_rate * 0.4, 0.02),
        outlier_rate=max(error_rate, 0.05),
        inconsistency_rate=max(error_rate * 0.6, 0.03),
    )
    # ~50 readings per device so device-level FD injection gives the expected rate
    n_devices = max(200, n // 50)
    dirty, clean, fd_rules = gen.generate(n, n_devices=n_devices, config=cfg)

    gt = compute_quality_profile(dirty, fd_rules)
    gt.n_records = len(dirty)

    cache.parent.mkdir(parents=True, exist_ok=True)
    dirty.to_parquet(cache, index=False)
    _save_gt_json(gt_path, gt)
    return dirty, gt, fd_rules


def _load_d7_real(data_path, n_records, seed, error_rate, rho, force_regen):
    """D7-real: Real-world IoT dataset.

    Looks for any of the following files (in order of priority):
      data/D7_iot_real.parquet          — pre-processed cache (any source)
      data/D7_intel_lab.parquet         — Intel Berkeley Lab sensor data
      data/D7_intel_lab.txt / .dat      — raw Intel Lab text file (auto-parsed)

    Intel Berkeley Lab data (2.3M readings, 54 sensors):
      Columns: date, time, epoch, moteid, temperature, humidity, light, voltage
      Download: http://db.csail.mit.edu/labdata/labdata.html
      Script:   outputs/paper_ready/scripts/download_d9_intel_lab.py

    Column mapping after loading:
      temperature → temperature_c, humidity → humidity_pct,
      light → light_lux, voltage → voltage_v
      Derived: device_id (from moteid), location_zone (moteid range → zone),
               alert_level (from threshold rules)
    FD rule: device_id → location_zone.

    Falls back to D7-synth with a warning if no file is found.
    """
    # Priority order for cache files
    candidates = [
        data_path / "D7_iot_real.parquet",
        data_path / "D7_intel_lab.parquet",
    ]
    raw_candidates = [
        data_path / "D7_intel_lab.txt",
        data_path / "D7_intel_lab.dat",
        data_path / "data.txt",       # Intel Lab default filename (uncompressed)
        data_path / "data.txt.gz",    # Intel Lab default filename (compressed)
        data_path / "D7_intel_lab.txt.gz",
    ]

    # Try pre-processed parquet cache first
    for cache in candidates:
        if cache.exists():
            df = pd.read_parquet(cache)
            if n_records and len(df) > n_records:
                df = df.sample(n=n_records, random_state=seed).reset_index(drop=True)
            fd_rules = {"device_id": "location_zone"} if "device_id" in df.columns else {}
            return df, None, fd_rules

    # Try raw Intel Lab text format
    for raw in raw_candidates:
        if raw.exists():
            df = _parse_intel_lab_raw(raw, seed)
            parquet_out = data_path / "D7_intel_lab.parquet"
            parquet_out.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(parquet_out, index=False)
            print(f"[D7-real] Parsed {len(df):,} rows from {raw.name} → cached to {parquet_out.name}")
            if n_records and len(df) > n_records:
                df = df.sample(n=n_records, random_state=seed).reset_index(drop=True)
            return df, None, {"device_id": "location_zone"}

    import warnings
    warnings.warn(
        "D7-real: No IoT dataset found in data/. "
        "Expected D7_iot_real.parquet or D7_intel_lab.parquet. "
        "Run outputs/paper_ready/scripts/download_d9_intel_lab.py to fetch the real data. "
        "Falling back to D7-synth.",
        stacklevel=3,
    )
    return _load_d7_synth(data_path, n_records, seed, error_rate, rho, force_regen)


def _parse_intel_lab_raw(path: Path, seed: int = 42) -> pd.DataFrame:
    """Parse the Intel Berkeley Lab sensor data file (plain text or .gz).

    Expected format (space-separated, no header):
        2004-02-28 00:59:58.471 2 1 122.153 -3.9397 11 2.03
        date       time          epoch moteid temp   hum   light voltage

    Returns a DataFrame with standardised column names matching D7 schema.
    """
    import gzip as _gzip, io as _io

    cols = ["date", "time", "epoch", "moteid",
            "temperature_c", "humidity_pct", "light_lux", "voltage_v"]

    if str(path).endswith(".gz"):
        with _gzip.open(path, "rb") as fh:
            raw_bytes = fh.read()
        file_obj = _io.StringIO(raw_bytes.decode("utf-8", errors="replace"))
    else:
        file_obj = path

    try:
        df = pd.read_csv(
            file_obj, sep=r"\s+", header=None, names=cols,
            na_values=["", "NA", "nan", "NaN"],
            on_bad_lines="skip",
        )
    except TypeError:
        df = pd.read_csv(
            file_obj, sep=r"\s+", header=None, names=cols,
            na_values=["", "NA", "nan", "NaN"],
            error_bad_lines=False,
        )

    # Drop parsing artifacts
    df = df.dropna(subset=["moteid", "temperature_c"]).copy()
    df["moteid"] = pd.to_numeric(df["moteid"], errors="coerce")
    df = df.dropna(subset=["moteid"])
    df["moteid"] = df["moteid"].astype(int)

    # Derived columns to match D7 schema
    n_zones = 8
    zone_labels = [f"Zone-{chr(65+i)}" for i in range(n_zones)]
    df["device_id"] = df["moteid"].apply(lambda m: f"dev_{int(m):04d}")
    df["location_zone"] = df["moteid"].apply(
        lambda m: zone_labels[int(m) % n_zones]
    )
    df["alert_level"] = np.where(
        (df["temperature_c"] > 35) | (df["voltage_v"] < 2.5),
        np.where(df["temperature_c"] > 45, "critical", "warning"),
        "normal",
    )

    # Add placeholder columns present in D7-synth but absent in Intel Lab
    rng = np.random.default_rng(seed)
    n = len(df)
    df["pressure_hpa"] = rng.normal(1013.25, 5.0, size=n)
    df["co2_ppm"]      = np.clip(rng.normal(800.0, 150.0, size=n), 300.0, 5000.0)
    df["current_ma"]   = np.clip(rng.normal(15.0, 3.0, size=n), 0.0, 100.0)
    df["battery_pct"]  = np.clip(rng.normal(60.0, 20.0, size=n), 0.0, 100.0)

    # Keep only D7 schema columns in canonical order
    keep = ["temperature_c", "humidity_pct", "pressure_hpa", "co2_ppm",
            "light_lux", "voltage_v", "current_ma", "battery_pct",
            "device_id", "location_zone", "alert_level"]
    return df[[c for c in keep if c in df.columns]].reset_index(drop=True)


def _load_gt_json(path: Path, n: int) -> Optional[QualityProfile]:
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    return QualityProfile(
        missing_rate=d["missing_rate"],
        duplicate_rate=d["duplicate_rate"],
        outlier_rate=d["outlier_rate"],
        inconsistency_rate=d["inconsistency_rate"],
        n_records=n,
        sample_size=n,
    )


def _save_gt_json(path: Path, gt: QualityProfile) -> None:
    with open(path, "w") as f:
        json.dump(gt.to_dict(), f, indent=2)
