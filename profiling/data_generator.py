"""Synthetic dataset generation: numpy-based (controlled errors) + SDV-based (scaling)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ErrorConfig:
    missing_rate: float = 0.05
    duplicate_rate: float = 0.02
    outlier_rate: float = 0.03
    inconsistency_rate: float = 0.02
    correlation_rho: float = 0.0  # injected attribute correlation


class SyntheticTabularGenerator:
    """Generate controlled synthetic tabular datasets with injected errors.

    Used for E4 (error-rate sweep) and E2 (correlation ablation).
    """

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng()

    def generate(
        self,
        n_records: int,
        n_cols: int = 8,
        config: Optional[ErrorConfig] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
        """Return (dirty_df, clean_df, fd_rules).

        dirty_df: dataset with injected errors
        clean_df: ground-truth error-free version
        fd_rules: dict{determinant→dependent} for inconsistency detection
        """
        cfg = config or ErrorConfig()

        # Generate clean base data with optional injected correlation
        if cfg.correlation_rho > 0:
            clean_df = self._generate_correlated(n_records, n_cols, cfg.correlation_rho)
        else:
            clean_df = self._generate_independent(n_records, n_cols)

        dirty_df = clean_df.copy()
        fd_rules: Dict[str, str] = {}

        # Inject missing values
        if cfg.missing_rate > 0:
            n_missing = int(n_records * n_cols * cfg.missing_rate)
            rows = self.rng.integers(0, n_records, size=n_missing)
            cols = self.rng.integers(0, n_cols, size=n_missing)
            for r, c in zip(rows, cols):
                dirty_df.iat[r, c] = np.nan

        # Inject duplicates
        if cfg.duplicate_rate > 0:
            n_dupes = int(n_records * cfg.duplicate_rate)
            src_rows = self.rng.integers(0, n_records, size=n_dupes)
            dupe_rows = dirty_df.iloc[src_rows].copy()
            dupe_rows.index = range(n_records, n_records + n_dupes)
            dirty_df = pd.concat([dirty_df, dupe_rows], ignore_index=True)

        # Inject outliers (values ×10 outside normal range)
        if cfg.outlier_rate > 0:
            n_outliers = int(n_records * cfg.outlier_rate)
            rows = self.rng.integers(0, n_records, size=n_outliers)
            cols = self.rng.integers(0, n_cols, size=n_outliers)
            for r, c in zip(rows, cols):
                col_name = dirty_df.columns[c]
                std = float(clean_df[col_name].std()) + 1e-6
                dirty_df.iat[r, c] = float(dirty_df.iat[r, c] or 0) + self.rng.choice([-1, 1]) * 10 * std

        # Inject inconsistencies (FD violations: col0 → col1)
        if cfg.inconsistency_rate > 0 and n_cols >= 2:
            det_col, dep_col = "col_0", "col_1"
            fd_rules[det_col] = dep_col
            n_incons = int(n_records * cfg.inconsistency_rate)
            rows = self.rng.integers(0, n_records, size=n_incons)
            for r in rows:
                dirty_df.at[r, dep_col] = float(self.rng.uniform(-100, 100))

        return dirty_df, clean_df, fd_rules

    def _generate_independent(self, n: int, n_cols: int) -> pd.DataFrame:
        data = {f"col_{i}": self.rng.normal(0, 1, size=n) for i in range(n_cols)}
        return pd.DataFrame(data)

    def _generate_correlated(self, n: int, n_cols: int, rho: float) -> pd.DataFrame:
        """Generate data with uniform pairwise correlation ρ using Cholesky decomposition."""
        cov = np.full((n_cols, n_cols), rho)
        np.fill_diagonal(cov, 1.0)
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            # Fallback: clip rho to make matrix PD
            cov = np.full((n_cols, n_cols), min(rho, 0.99))
            np.fill_diagonal(cov, 1.0)
            L = np.linalg.cholesky(cov)
        z = self.rng.normal(size=(n, n_cols))
        data = z @ L.T
        return pd.DataFrame(data, columns=[f"col_{i}" for i in range(n_cols)])


# ──────────────────────────────────────────────────────────────────
# Error injection for pre-generated DataFrames (E5 Outsourcing data)
# ──────────────────────────────────────────────────────────────────

def inject_errors_into_df(
    df: pd.DataFrame,
    rng: np.random.Generator,
    error_rate: float = 0.05,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Inject controlled errors into an existing mixed-type DataFrame.

    Works on Adult census schema (numeric + categorical columns).
    Returns (dirty_df, fd_rules). The dirty_df may be larger than df if
    duplicate_rate > 0 (duplicated rows are appended).
    """
    n = len(df)
    dirty = df.copy()
    fd_rules: Dict[str, str] = {}

    numeric_cols = dirty.select_dtypes(include=[np.number]).columns.tolist()

    # Missing: vectorized per-column bernoulli
    for col in dirty.columns:
        mask = rng.random(n) < error_rate
        dirty.loc[mask, col] = np.nan

    # Outliers on numeric columns (×10 std from mean)
    n_out_per_col = max(1, int(n * error_rate * 0.5 / max(len(numeric_cols), 1)))
    for col in numeric_cols:
        finite = dirty[col].dropna().values
        if len(finite) < 4:
            continue
        mean_v = float(np.nanmean(finite))
        std_v = float(np.nanstd(finite)) + 1e-6
        rows = rng.integers(0, n, size=n_out_per_col)
        signs = rng.choice(np.array([-1.0, 1.0]), size=n_out_per_col)
        dirty.iloc[rows, dirty.columns.get_loc(col)] = mean_v + signs * 10 * std_v

    # Duplicates: append copies (GT computation will see them)
    n_dupes = int(n * error_rate * 0.5)
    if n_dupes > 0:
        src_idx = rng.integers(0, n, size=n_dupes)
        dupe_rows = df.iloc[src_idx].copy().reset_index(drop=True)
        dirty = pd.concat([dirty, dupe_rows], ignore_index=True)

    # No FD injection for pre-generated Outsourcing data: SDV GaussianCopula destroys
    # categorical-to-numeric correspondences (produces 400+ unique "education" strings),
    # so the education → education-num FD is already broken in the clean source data.
    # inconsistency_rate will be 0 in GT; profilers estimate 0 trivially.

    return dirty, fd_rules


# ──────────────────────────────────────────────────────────────────
# IoT sensor data generator (D7-synth)
# ──────────────────────────────────────────────────────────────────

class IoTSensorGenerator:
    """Generate synthetic IoT sensor data with realistic distributions and quality defects.

    Schema (11 columns):
      Numeric (8):  temperature_c, humidity_pct, pressure_hpa, co2_ppm,
                    light_lux, voltage_v, current_ma, battery_pct
      Categorical (3): device_id, location_zone, alert_level

    Quality-defect profile is intentionally numeric-dominant so the IQR proxy
    can detect the injected outliers. This contrasts with D2/D3/D4 (admin/census)
    where defects concentrate in categorical columns.

    FD rule: device_id → location_zone (violated by ``inconsistency_rate`` fraction).
    """

    # Column correlations: Cholesky-factored 8×8 numeric covariance
    # temp↔hum: -0.35, temp↔pressure: 0.20, voltage↔battery: 0.70, rest: ~0
    _CORR = np.array([
        [ 1.00, -0.35,  0.20,  0.10,  0.05,  0.00,  0.00,  0.00],  # temperature
        [-0.35,  1.00, -0.10,  0.05, -0.05,  0.00,  0.00,  0.00],  # humidity
        [ 0.20, -0.10,  1.00,  0.05,  0.00,  0.00,  0.00,  0.00],  # pressure
        [ 0.10,  0.05,  0.05,  1.00,  0.10,  0.00,  0.00,  0.00],  # co2
        [ 0.05, -0.05,  0.00,  0.10,  1.00,  0.00,  0.00,  0.00],  # light
        [ 0.00,  0.00,  0.00,  0.00,  0.00,  1.00,  0.30,  0.70],  # voltage
        [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.30,  1.00,  0.40],  # current
        [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.70,  0.40,  1.00],  # battery
    ])

    _ZONES = ["Zone-A", "Zone-B", "Zone-C", "Zone-D",
              "Zone-E", "Zone-F", "Zone-G", "Zone-H"]

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng()

    def generate(
        self,
        n_records: int,
        n_devices: int = 200,
        config: Optional[ErrorConfig] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
        """Return (dirty_df, clean_df, fd_rules).

        Inconsistency injection is device-level: ``inconsistency_rate`` fraction
        of all device_ids get one row's location_zone flipped to a wrong zone.
        Because compute_quality_profile flags ALL rows of a device once that device
        appears in two zones, the measured rate ≈ inconsistency_rate × (n_records/n_records) = inconsistency_rate,
        provided n_devices ≈ n_records / rows_per_device (set by caller).
        """
        cfg = config or ErrorConfig(
            missing_rate=0.08,
            duplicate_rate=0.02,
            outlier_rate=0.05,
            inconsistency_rate=0.03,
        )

        clean_df = self._generate_clean(n_records, n_devices)
        dirty_df = clean_df.copy()

        fd_rules: Dict[str, str] = {"device_id": "location_zone"}

        numeric_cols = ["temperature_c", "humidity_pct", "pressure_hpa", "co2_ppm",
                        "light_lux", "voltage_v", "current_ma", "battery_pct"]

        # Missing: sensor dropouts in numeric cols only (vectorised, no double-injection)
        n_total_cells = n_records * len(numeric_cols)
        n_missing = int(n_total_cells * cfg.missing_rate)
        if n_missing > 0:
            miss_rows = self.rng.integers(0, n_records, size=n_missing)
            miss_col_idx = self.rng.integers(0, len(numeric_cols), size=n_missing)
            for col_i, col in enumerate(numeric_cols):
                mask = miss_col_idx == col_i
                rows_for_col = miss_rows[mask]
                if rows_for_col.size:
                    dirty_df.iloc[rows_for_col, dirty_df.columns.get_loc(col)] = np.nan

        # Outliers: sensor malfunctions → extreme values in IQR-visible numeric cols
        n_outliers = int(n_records * cfg.outlier_rate)
        if n_outliers > 0:
            out_rows = self.rng.integers(0, n_records, size=n_outliers)
            out_col_names = self.rng.choice(
                ["temperature_c", "co2_ppm", "voltage_v"], size=n_outliers,
            )
            outlier_specs = {
                "temperature_c": (22.0,   3.0,  15.0),
                "co2_ppm":       (800.0, 150.0, 12.0),
                "voltage_v":     (3.7,    0.1,  20.0),
            }
            for r, col in zip(out_rows, out_col_names):
                mu, sigma, mult = outlier_specs[col]
                sign = float(self.rng.choice([-1.0, 1.0]))
                dirty_df.iat[int(r), dirty_df.columns.get_loc(col)] = mu + sign * mult * sigma

        # Duplicates: packet buffering
        n_dupes = int(n_records * cfg.duplicate_rate)
        if n_dupes > 0:
            src = self.rng.integers(0, n_records, size=n_dupes)
            extra = clean_df.iloc[src].copy().reset_index(drop=True)
            dirty_df = pd.concat([dirty_df, extra], ignore_index=True)

        # Inconsistencies: device-level FD violation (device_id → location_zone).
        # Select inconsistency_rate fraction of all devices; flip ONE row's zone for each.
        # compute_quality_profile will flag all rows of a device that maps to 2+ zones,
        # so measured rate ≈ (n_violating_devices / n_devices) ≈ inconsistency_rate.
        all_devices = dirty_df["device_id"].unique()
        n_violating = max(1, int(len(all_devices) * cfg.inconsistency_rate))
        violating_devices = self.rng.choice(all_devices, size=n_violating, replace=False)
        for dev in violating_devices:
            dev_idx = dirty_df.index[dirty_df["device_id"] == dev].tolist()
            if not dev_idx:
                continue
            r = int(self.rng.choice(dev_idx))
            current_zone = dirty_df.at[r, "location_zone"]
            wrong_zone = self.rng.choice(
                [z for z in self._ZONES if z != current_zone]
            )
            dirty_df.at[r, "location_zone"] = wrong_zone

        return dirty_df, clean_df, fd_rules

    def _generate_clean(self, n: int, n_devices: int) -> pd.DataFrame:
        # Draw correlated standard normals via Cholesky
        L = np.linalg.cholesky(self._CORR)
        z = self.rng.normal(size=(n, 8)) @ L.T

        # Transform to physical ranges
        temp   = z[:, 0] * 3.0 + 22.0
        hum    = np.clip(z[:, 1] * 10.0 + 55.0, 0.0, 100.0)
        press  = z[:, 2] * 5.0 + 1013.25
        co2    = np.clip(z[:, 3] * 150.0 + 800.0, 300.0, 5000.0)
        light  = np.clip(np.exp(z[:, 4] * 1.2 + 5.0), 0.0, 100_000.0)
        volt   = np.clip(z[:, 5] * 0.1 + 3.7, 2.0, 4.5)
        curr   = np.clip(z[:, 6] * 3.0 + 15.0, 0.0, 100.0)
        batt   = np.clip(z[:, 7] * 20.0 + 60.0, 0.0, 100.0)

        # Categorical: device_id and zone are fixed per reading; zone FD holds in clean data
        devices = [f"dev_{i+1:04d}" for i in range(n_devices)]
        dev_ids = self.rng.choice(devices, size=n)
        # Assign each device to exactly one zone (stable FD)
        dev_zone_map = {d: self._ZONES[i % len(self._ZONES)] for i, d in enumerate(devices)}
        zones = np.array([dev_zone_map[d] for d in dev_ids])

        # alert_level based on actual sensor readings
        alert = np.where(
            (temp > 35) | (co2 > 2000) | (volt < 2.5),
            np.where((temp > 45) | (co2 > 3500), "critical", "warning"),
            "normal",
        )

        return pd.DataFrame({
            "temperature_c": temp,
            "humidity_pct":  hum,
            "pressure_hpa":  press,
            "co2_ppm":       co2,
            "light_lux":     light,
            "voltage_v":     volt,
            "current_ma":    curr,
            "battery_pct":   batt,
            "device_id":     dev_ids,
            "location_zone": zones,
            "alert_level":   alert,
        })


# ──────────────────────────────────────────────────────────────────
# SDV-based synthetic generator for scaling experiments (E5)
# ──────────────────────────────────────────────────────────────────

def generate_sdv_dataset(
    n_rows: int,
    n_cols: int = 10,
    sdv_model: str = "GaussianCopula",
    seed: int = 42,
    error_rate: float = 0.05,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """Generate a realistic synthetic dataset using SDV, then inject errors.

    Falls back to numpy-based generation if SDV is not installed.
    """
    try:
        from sdv.metadata import SingleTableMetadata
        from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer

        # Build a small base table (10k rows) and fit the model
        _base_n = min(n_rows, 10_000)
        rng = np.random.default_rng(seed)
        base_data: Dict[str, np.ndarray] = {}
        for i in range(n_cols):
            base_data[f"col_{i}"] = rng.normal(i * 0.5, 1.0, size=_base_n)
        # Add a categorical column to make data more realistic
        base_data["category"] = rng.choice(["A", "B", "C", "D"], size=_base_n)
        base_df = pd.DataFrame(base_data)

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(base_df)

        if sdv_model == "CTGAN":
            synthesizer = CTGANSynthesizer(metadata, epochs=50)
        else:
            synthesizer = GaussianCopulaSynthesizer(metadata)

        synthesizer.fit(base_df)
        clean_df = synthesizer.sample(num_rows=n_rows)
        clean_df = clean_df.reset_index(drop=True)

    except ImportError:
        # SDV not installed — fall back to correlated numpy data
        import warnings
        warnings.warn("SDV not installed; falling back to numpy-based correlated data.", stacklevel=2)
        gen = SyntheticTabularGenerator(rng=np.random.default_rng(seed))
        clean_df, _, _ = gen.generate(n_rows, n_cols=n_cols, config=ErrorConfig(
            missing_rate=0, duplicate_rate=0, outlier_rate=0, inconsistency_rate=0,
            correlation_rho=0.3,
        ))

    # Inject errors on top of clean data
    gen2 = SyntheticTabularGenerator(rng=np.random.default_rng(seed + 1))
    cfg = ErrorConfig(
        missing_rate=error_rate,
        duplicate_rate=error_rate / 2,
        outlier_rate=error_rate / 2,
        inconsistency_rate=error_rate / 4,
    )
    # We inject directly on the SDV-generated clean_df
    numeric_cols = clean_df.select_dtypes(include=[np.number]).columns.tolist()
    n = len(clean_df)
    dirty_df = clean_df.copy()
    rng2 = np.random.default_rng(seed + 2)

    n_missing = int(n * len(numeric_cols) * cfg.missing_rate)
    if n_missing > 0:
        r = rng2.integers(0, n, size=n_missing)
        c = rng2.integers(0, len(numeric_cols), size=n_missing)
        for ri, ci in zip(r, c):
            dirty_df.iat[ri, ci] = np.nan

    n_dupes = int(n * cfg.duplicate_rate)
    if n_dupes > 0:
        src = rng2.integers(0, n, size=n_dupes)
        extra = clean_df.iloc[src].copy().reset_index(drop=True)
        extra.index = range(n, n + n_dupes)
        dirty_df = pd.concat([dirty_df, extra], ignore_index=True)

    fd_rules: Dict[str, str] = {}
    if len(numeric_cols) >= 2:
        fd_rules[numeric_cols[0]] = numeric_cols[1]

    return dirty_df, clean_df, fd_rules
