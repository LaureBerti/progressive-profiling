"""Quality profile estimation from a dataset sample."""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class QualityIndicator(str, Enum):
    MISSING = "missing_rate"
    DUPLICATE = "duplicate_rate"
    OUTLIER = "outlier_rate"
    INCONSISTENCY = "inconsistency_rate"


@dataclass
class QualityProfile:
    missing_rate: float
    duplicate_rate: float
    outlier_rate: float
    inconsistency_rate: float
    n_records: int
    sample_size: int
    wall_time_s: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "missing_rate": self.missing_rate,
            "duplicate_rate": self.duplicate_rate,
            "outlier_rate": self.outlier_rate,
            "inconsistency_rate": self.inconsistency_rate,
        }

    def relative_error(self, ground_truth: "QualityProfile") -> Dict[str, float]:
        """Compute bounded relative error per indicator.

        Denominator floor = max(gt, 0.05): prevents blowup when GT is near zero
        while keeping the metric meaningful for typical quality rates.
        All per-indicator errors are capped at 1.0 (100%).
        """
        gt = ground_truth.to_dict()
        est = self.to_dict()
        errors: Dict[str, float] = {}
        for key in gt:
            gt_val = gt[key]
            est_val = est[key]
            denom = max(gt_val, 0.05)
            errors[key] = min(abs(est_val - gt_val) / denom, 1.0)
        errors["mean"] = float(np.mean(list(errors.values())))
        errors["abs_mean"] = float(np.mean([abs(est[k] - gt[k]) for k in gt]))
        return errors


def compute_quality_profile(
    df: pd.DataFrame,
    fd_rules: Optional[Dict[str, str]] = None,
    outlier_z_thresh: float = 3.0,
    sample_weights: Optional[np.ndarray] = None,
) -> QualityProfile:
    """Estimate all four quality indicators from a DataFrame.

    fd_rules: dict mapping determinant column to dependent column.
    sample_weights: importance weights (shape = n rows, must sum > 0).
        When provided, Horvitz-Thompson weighted estimates are used for all
        indicators, correcting for sampling bias introduced by MCMC samplers.
        Weights should be proportional to 1/sampling_probability per row.
    """
    n = len(df)
    if n == 0:
        return QualityProfile(0.0, 0.0, 0.0, 0.0, 0, 0)

    # Normalize weights once; fall back to uniform when not provided
    if sample_weights is not None:
        w = np.asarray(sample_weights, dtype=float)
        w = w / (w.sum() + 1e-12)
    else:
        w = None

    # ── Missing rate ──────────────────────────────────────────────────────────
    if w is not None:
        row_miss = df.isnull().mean(axis=1).values
        missing_rate = float(np.dot(row_miss, w))
    else:
        total_cells = df.size
        missing = int(df.isnull().sum().sum())
        missing_rate = missing / total_cells if total_cells > 0 else 0.0

    # ── Duplicate rate ────────────────────────────────────────────────────────
    dup_mask = df.duplicated(keep=False).values.astype(float)
    if w is not None:
        duplicate_rate = float(np.dot(dup_mask, w))
    else:
        duplicate_rate = float(dup_mask.mean())

    # ── Outlier rate (IQR-based on numeric columns) ───────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_mask = np.zeros(n, dtype=bool)
    if numeric_cols:
        for col in numeric_cols:
            col_vals = df[col].dropna()
            if len(col_vals) < 4:
                continue
            q1, q3 = float(col_vals.quantile(0.25)), float(col_vals.quantile(0.75))
            iqr = q3 - q1
            if iqr == 0:
                mu, sigma = float(col_vals.mean()), float(col_vals.std()) + 1e-9
                outlier_mask |= (df[col].notna() & (np.abs((df[col] - mu) / sigma) > outlier_z_thresh)).values
            else:
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outlier_mask |= ((df[col] < lo) | (df[col] > hi)).values
    if w is not None:
        outlier_rate = float(np.dot(outlier_mask.astype(float), w))
    else:
        outlier_rate = float(outlier_mask.mean())

    # ── Inconsistency rate (functional dependency violations) ─────────────────
    incons_mask = np.zeros(n, dtype=bool)
    if fd_rules:
        for det_col, dep_col in fd_rules.items():
            if det_col not in df.columns or dep_col not in df.columns:
                continue
            multi = df.groupby(det_col)[dep_col].nunique()
            violating_keys = multi[multi > 1].index
            incons_mask |= df[det_col].isin(violating_keys).values
    if w is not None:
        inconsistency_rate = float(np.dot(incons_mask.astype(float), w))
    else:
        inconsistency_rate = float(incons_mask.mean())

    return QualityProfile(
        missing_rate=missing_rate,
        duplicate_rate=duplicate_rate,
        outlier_rate=outlier_rate,
        inconsistency_rate=inconsistency_rate,
        n_records=n,
        sample_size=n,
    )
