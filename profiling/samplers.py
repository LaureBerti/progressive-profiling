"""Sampling strategies: static baselines + MCMC family + DAG-enhanced.

Performance contract: compute_quality_profile() is called ONCE per condition,
at the end of sampling. MCMC samplers pass importance weights to correct for
sampling bias (Horvitz-Thompson estimation): w_i ∝ 1/score_i for sampled rows.
"""
from __future__ import annotations

import time
import warnings
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .quality_profile import QualityProfile, compute_quality_profile
from .dag import AttributeDAG


@dataclass
class SamplingResult:
    profile: QualityProfile
    n_sampled: int
    time_s: float
    method: str
    n_steps: int = 0
    acceptance_rate: float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Shared pre-computation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _precompute_error_proxy(df: pd.DataFrame) -> np.ndarray:
    """Return an error-likelihood score per row (fully vectorized, O(N×M)).

    Uses IQR-based outlier detection matching compute_quality_profile so the
    proxy aligns with the measurement — this reduces importance-weight bias.
    Score = fraction of outlier cols per row + missing fraction per row.
    Vectorized over all columns simultaneously; no Python loop.
    """
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return np.zeros(len(df))

    arr = numeric.values.astype(float)  # (N, M)
    missing = np.isnan(arr)

    # Vectorized IQR outlier detection — all columns at once
    # suppress all-NaN column warnings (e.g. float columns that are entirely null)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        q1 = np.nanpercentile(arr, 25, axis=0)   # (M,)
        q3 = np.nanpercentile(arr, 75, axis=0)   # (M,)
    iqr = q3 - q1                             # (M,)

    # For columns with non-zero IQR: Tukey fences
    nonzero_iqr = iqr > 0
    lo = np.where(nonzero_iqr, q1 - 1.5 * iqr, -np.inf)  # (M,)
    hi = np.where(nonzero_iqr, q3 + 1.5 * iqr,  np.inf)  # (M,)
    iqr_outlier = (arr < lo[None, :]) | (arr > hi[None, :])  # (N, M)

    # For zero-IQR columns: z-score fallback (rare)
    # suppress warnings for all-NaN columns (e.g. string cols cast to float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        col_means = np.nanmean(arr, axis=0)
        col_stds  = np.nanstd(arr, axis=0) + 1e-9
    z = np.abs((arr - col_means[None, :]) / col_stds[None, :])
    z_outlier = (z > 3.0) & ~nonzero_iqr[None, :]

    outlier_mask = (iqr_outlier | z_outlier) & ~missing
    outlier_score = outlier_mask.mean(axis=1)
    missing_score = missing.mean(axis=1)
    return (outlier_score + missing_score).astype(float)


class BaseSampler(ABC):
    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng()

    @abstractmethod
    def sample(
        self,
        df: pd.DataFrame,
        budget_fraction: float = 0.5,
        fd_rules: Optional[Dict[str, str]] = None,
    ) -> SamplingResult:
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Static baselines
# ─────────────────────────────────────────────────────────────────────────────

class RandomUniformSampler(BaseSampler):
    def sample(self, df, budget_fraction=0.5, fd_rules=None):
        t0 = time.time()
        n = max(1, int(len(df) * budget_fraction))
        idx = self.rng.choice(len(df), size=n, replace=False)
        profile = compute_quality_profile(df.iloc[idx], fd_rules)
        profile.n_records = len(df)
        return SamplingResult(profile=profile, n_sampled=n,
                              time_s=time.time() - t0, method="random_uniform")


class GeometricSampler(BaseSampler):
    """Geometrically decreasing batch sizes: N/2, N/4, N/8 …"""

    def sample(self, df, budget_fraction=0.5, fd_rules=None):
        t0 = time.time()
        N = len(df)
        budget = int(N * budget_fraction)
        mask = np.zeros(N, dtype=bool)
        batch = N // 2
        while batch >= 1 and mask.sum() < budget:
            avail = np.where(~mask)[0]
            take = min(batch, budget - int(mask.sum()), len(avail))
            chosen = self.rng.choice(avail, size=take, replace=False)
            mask[chosen] = True
            batch //= 2

        idx = np.where(mask)[0]
        profile = compute_quality_profile(df.iloc[idx], fd_rules)
        profile.n_records = N
        return SamplingResult(profile=profile, n_sampled=int(mask.sum()),
                              time_s=time.time() - t0, method="geometric")


class YamaneSampler(BaseSampler):
    """Yamane formula: n = N / (1 + N·e²)."""

    def __init__(self, margin_of_error: float = 0.05, rng=None):
        super().__init__(rng)
        self.e = margin_of_error

    def sample(self, df, budget_fraction=0.5, fd_rules=None):
        t0 = time.time()
        N = len(df)
        n = max(1, int(N / (1 + N * self.e ** 2)))
        idx = self.rng.choice(N, size=min(n, N), replace=False)
        profile = compute_quality_profile(df.iloc[idx], fd_rules)
        profile.n_records = N
        return SamplingResult(profile=profile, n_sampled=len(idx),
                              time_s=time.time() - t0, method="yamane")


# ─────────────────────────────────────────────────────────────────────────────
# MCMC family — vectorized, single quality-profile call at end
# ─────────────────────────────────────────────────────────────────────────────

class _MCMCSampler(BaseSampler):
    """Base MCMC sampler. Uses a precomputed error-proxy score for acceptance.

    Key performance contract: compute_quality_profile() is called ONCE
    after all sampling steps are done. The acceptance criterion is a
    cheap vectorized proxy derived from missing values + z-scores.
    """

    def __init__(self, batch_size: int = 200, rng=None, uniform_weights: bool = False):
        super().__init__(rng)
        self.batch_size = batch_size
        self.uniform_weights = uniform_weights

    # ── subclass interface ──────────────────────────────────────────────────

    def _propose(self, unsampled_idx: np.ndarray, scores: np.ndarray,
                 df: pd.DataFrame) -> np.ndarray:
        """Return indices to add to the sample (subset of unsampled_idx)."""
        # Default: uniform over unsampled
        take = min(self.batch_size, len(unsampled_idx))
        return self.rng.choice(unsampled_idx, size=take, replace=False)

    def _accept(self, curr_score: float, prop_score: float) -> bool:
        return True

    def _method_name(self) -> str:
        return "mcmc"

    # ── shared loop ─────────────────────────────────────────────────────────

    def sample(self, df, budget_fraction=0.5, fd_rules=None):
        t0 = time.time()
        N = len(df)
        budget = max(self.batch_size, int(N * budget_fraction))

        # Pre-compute per-row error proxy (vectorized, done once)
        scores = _precompute_error_proxy(df)
        # Cache statistics for _propose() to avoid per-step recomputation
        self._score_thresh = float(np.median(scores))
        self._scores_uniform = bool(float(scores.max()) == float(scores.min()))

        # Warm start: bias toward high-score rows
        init_n = min(self.batch_size, N)
        probs = scores + 1e-6
        probs /= probs.sum()
        init_idx = self.rng.choice(N, size=init_n, replace=False, p=probs)

        sampled = np.zeros(N, dtype=bool)
        sampled[init_idx] = True
        n_sampled = int(init_n)
        # Incremental running mean — avoids O(N) full-scan per step
        curr_score = float(scores[init_idx].mean())

        # Pool: dense array of unsampled global indices with O(k) removal.
        # pool_pos[i] = position of global index i in pool (or N if removed).
        pool = np.setdiff1d(np.arange(N, dtype=np.int64), init_idx)
        pool_size = len(pool)
        pool_pos = np.full(N, N, dtype=np.int64)  # sentinel = N means "sampled"
        pool_pos[pool] = np.arange(pool_size, dtype=np.int64)

        accepts = steps = 0

        while n_sampled < budget:
            if pool_size == 0:
                break

            unsampled_idx = pool[:pool_size]
            proposed_idx = self._propose(unsampled_idx, scores, df)
            prop_score = float(scores[proposed_idx].mean())

            if self._accept(curr_score, prop_score):
                add_idx = proposed_idx
                accepts += 1
            else:
                # Force progress: add a small random batch so we don't stall
                force_n = max(1, min(self.batch_size // 4, pool_size))
                add_idx = self.rng.choice(pool[:pool_size], size=force_n, replace=False)
            steps += 1

            # Remove add_idx from pool (swap-to-end, descending position order)
            positions = pool_pos[add_idx]
            order = np.argsort(positions)[::-1]
            for k in order:
                idx = int(add_idx[k])
                p = int(pool_pos[idx])
                if p >= pool_size:
                    continue
                pool_size -= 1
                if p < pool_size:
                    last = int(pool[pool_size])
                    pool[p] = last
                    pool_pos[last] = p
                pool_pos[idx] = N  # sentinel: removed

            n_new = len(add_idx)
            curr_score = (curr_score * n_sampled + float(scores[add_idx].sum())) / (n_sampled + n_new)
            n_sampled += n_new
            sampled[add_idx] = True

        # Single quality-profile call with importance weights (Horvitz-Thompson correction)
        # w_i ∝ 1/score_i corrects for the bias introduced by oversampling high-score rows.
        # uniform_weights=True retains the DAG proposal but disables IQR-based HT weighting
        # (used in E7a to isolate whether failure stems from the proxy or the graph structure).
        final_idx = np.where(sampled)[0][:budget]
        final_scores = scores[final_idx]
        if self.uniform_weights:
            iw = np.ones(len(final_idx))
        else:
            iw = 1.0 / (final_scores + 1e-6)
            # Clip at 95th percentile: score=0 rows otherwise get weight=1/1e-6=10^6,
            # dominating the estimate and zeroing out quality rates on real data.
            iw_cap = float(np.percentile(iw, 95))
            if iw_cap > 0:
                iw = np.minimum(iw, iw_cap)
        iw = iw / iw.sum()
        profile = compute_quality_profile(df.iloc[final_idx], fd_rules, sample_weights=iw)
        profile.n_records = N

        return SamplingResult(
            profile=profile,
            n_sampled=len(final_idx),
            time_s=time.time() - t0,
            method=self._method_name(),
            n_steps=steps,
            acceptance_rate=accepts / max(steps, 1),
        )


class MetropolisHastingsSampler(_MCMCSampler):
    """MH: accept if proposed score ≥ current (with temperature-scaled probability)."""

    def __init__(self, lam: float = 2.0, batch_size: int = 200, rng=None):
        super().__init__(batch_size, rng)
        self.lam = lam

    def _accept(self, curr_score, prop_score):
        log_ratio = self.lam * (prop_score - curr_score)
        return float(np.log(self.rng.uniform() + 1e-12)) < log_ratio

    def _method_name(self):
        return "metropolis_hastings"


class GibbsSampler(_MCMCSampler):
    """Gibbs: sample uniformly within a random quantile bin of one attribute.

    Always accepts. Ignores cross-attribute correlations — controlled ablation for C-A2.
    """

    def _propose(self, unsampled_idx, scores, df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            take = min(self.batch_size, len(unsampled_idx))
            return self.rng.choice(unsampled_idx, size=take, replace=False)

        col = self.rng.choice(numeric_cols)
        col_arr = df[col].values
        q_lo = self.rng.uniform(0, 0.9)
        lo = np.nanquantile(col_arr, q_lo)
        hi = np.nanquantile(col_arr, q_lo + 0.1)
        mask = (col_arr[unsampled_idx] >= lo) & (col_arr[unsampled_idx] <= hi)
        candidates = unsampled_idx[mask] if mask.any() else unsampled_idx
        take = min(self.batch_size, len(candidates))
        return self.rng.choice(candidates, size=take, replace=False)

    def _accept(self, curr_score, prop_score):
        return True  # Gibbs always accepts

    def _method_name(self):
        return "gibbs"


class DAGSampler(_MCMCSampler):
    """DAG-cognizant MH: uses attribute dependency DAG to bias proposals.

    Steers sampling toward attribute combinations correlated with minority
    error classes, using DAG cluster_priority() to pick the focal column.
    Uses the MH acceptance ratio (same as MetropolisHastingsSampler) but
    with a smarter proposal that the DAG informs.
    """

    def __init__(self, dag: Optional[AttributeDAG] = None, lam: float = 2.0,
                 batch_size: int = 200, rng=None, uniform_weights: bool = False):
        super().__init__(batch_size, rng, uniform_weights=uniform_weights)
        self.dag = dag
        self.lam = lam
        self._col_arrays: Optional[Dict[str, np.ndarray]] = None
        self._col_thresholds: Optional[Dict[str, Tuple[float, float]]] = None

    def _get_col_arrays(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        if self._col_arrays is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            self._col_arrays = {c: df[c].values for c in numeric_cols}
            # Precompute 5th/95th percentile thresholds once for the whole column.
            # Avoids O(N log N) sort inside the proposal loop at each MCMC step.
            self._col_thresholds = {}
            for c, arr in self._col_arrays.items():
                finite = arr[np.isfinite(arr)]
                if len(finite) >= 4:
                    self._col_thresholds[c] = (
                        float(np.percentile(finite, 5)),
                        float(np.percentile(finite, 95)),
                    )
        return self._col_arrays

    def _propose(self, unsampled_idx, scores, df):
        col_arrays = self._get_col_arrays(df)
        if not col_arrays or self.dag is None:
            take = min(self.batch_size, len(unsampled_idx))
            # Uniform scores: skip filtering (avoids O(pool) ops per step)
            if getattr(self, "_scores_uniform", False):
                return self.rng.choice(unsampled_idx, size=take, replace=False)
            # Non-uniform: prefer rows above global score median
            thresh = getattr(self, "_score_thresh", 0.0)
            sub_scores = scores[unsampled_idx]
            hi_mask = sub_scores >= thresh
            n_hi = int(hi_mask.sum())
            candidates = unsampled_idx[hi_mask] if n_hi >= take else unsampled_idx
            return self.rng.choice(candidates, size=take, replace=False)

        # Pick focal column using DAG priority
        dag_cols = [c for c in self.dag.columns if c in col_arrays]
        if not dag_cols:
            take = min(self.batch_size, len(unsampled_idx))
            return self.rng.choice(unsampled_idx, size=take, replace=False)

        priority = self.dag.cluster_priority()
        dag_col_idx = np.array([self.dag.col_index[c] for c in dag_cols])
        sub_prio = np.nan_to_num(priority[dag_col_idx], nan=0.0)
        total = sub_prio.sum()
        if total <= 0:
            take = min(self.batch_size, len(unsampled_idx))
            return self.rng.choice(unsampled_idx, size=take, replace=False)
        sub_prio = sub_prio / total
        pivot_col = self.rng.choice(dag_cols, p=sub_prio)

        # Use precomputed thresholds — O(pool_size) lookup, no sort at each step
        col_arr = col_arrays[pivot_col]
        sub_vals = col_arr[unsampled_idx]
        if pivot_col not in self._col_thresholds:
            take = min(self.batch_size, len(unsampled_idx))
            return self.rng.choice(unsampled_idx, size=take, replace=False)

        lo, hi = self._col_thresholds[pivot_col]
        extreme_mask = (sub_vals <= lo) | (sub_vals >= hi) | ~np.isfinite(sub_vals)
        candidates = unsampled_idx[extreme_mask] if extreme_mask.any() else unsampled_idx

        # Uniform choice within extreme candidates — O(1) alias table instead of O(|candidates|).
        # Extreme filtering already provides DAG guidance; p=-weighting adds negligible benefit.
        take = min(self.batch_size, len(candidates))
        return self.rng.choice(candidates, size=take, replace=False)

    def _accept(self, curr_score, prop_score):
        log_ratio = self.lam * (prop_score - curr_score)
        return float(np.log(self.rng.uniform() + 1e-12)) < log_ratio

    def _method_name(self):
        return "dag"


class DAGUniformWeightsSampler(DAGSampler):
    """DAG-guided sampler with uniform importance weights (ablation variant).

    Retains DAGSampler's graph-based pivot selection (column priority from
    AttributeDAG) but replaces IQR-derived HT weights with uniform weights.
    This isolates whether DAG's degradation on real datasets stems from the
    graph-based proposal or the IQR-based error proxy. Used in E7a.
    """

    def __init__(self, dag: Optional[AttributeDAG] = None, lam: float = 2.0,
                 batch_size: int = 200, rng=None):
        super().__init__(dag=dag, lam=lam, batch_size=batch_size, rng=rng,
                         uniform_weights=True)

    def _method_name(self) -> str:
        return "dag_uniform"


# ─────────────────────────────────────────────────────────────────────────────
# Stratified samplers
# ─────────────────────────────────────────────────────────────────────────────

class StratifiedColumnSampler(BaseSampler):
    """Stratify rows by column-type defect profile (numeric vs categorical).

    Three strata:
      S0 — numeric-defect rows: any IQR outlier or NaN in a numeric column.
      S1 — categorical-defect rows: any null in a string/object column.
      S2 — defect-free rows: neither.
    Overlap (S0 ∩ S1) is assigned to S0 (numeric stratum).

    Budget allocation: Neyman optimal — n_h ∝ N_h × std_h(proxy score).
    Uniform draw within strata; uniform HT weights.
    """

    def sample(self, df: pd.DataFrame, budget_fraction: float = 0.5,
               fd_rules=None) -> SamplingResult:
        t0 = time.time()
        N = len(df)
        n = max(1, int(N * budget_fraction))

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(exclude=[np.number]).columns

        # Numeric defect mask — vectorized IQR + NaN detection
        numeric_defect = np.zeros(N, dtype=bool)
        if len(numeric_cols) > 0:
            arr = df[numeric_cols].values.astype(float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                q1 = np.nanpercentile(arr, 25, axis=0)
                q3 = np.nanpercentile(arr, 75, axis=0)
            iqr = q3 - q1
            nonzero = iqr > 0
            lo = np.where(nonzero, q1 - 1.5 * iqr, -np.inf)
            hi = np.where(nonzero, q3 + 1.5 * iqr,  np.inf)
            numeric_defect = (
                ((arr < lo[None, :]) | (arr > hi[None, :])) | np.isnan(arr)
            ).any(axis=1)

        # Categorical defect mask
        cat_defect = np.zeros(N, dtype=bool)
        if len(cat_cols) > 0:
            cat_defect = df[cat_cols].isnull().values.any(axis=1)

        s0 = np.where(numeric_defect)[0]                          # includes overlap
        s1 = np.where(cat_defect & ~numeric_defect)[0]
        s2 = np.where(~numeric_defect & ~cat_defect)[0]
        strata = [s0, s1, s2]

        scores = _precompute_error_proxy(df)

        # Neyman allocation: n_h ∝ N_h × std_h
        Nh = np.array([len(s) for s in strata], dtype=float)
        Sh = np.array([float(scores[s].std()) if len(s) > 1 else 0.0
                       for s in strata])
        weights = Nh * Sh
        total = weights.sum()
        if total == 0:
            weights = Nh.copy()
            total = weights.sum()

        alloc = np.floor(n * weights / total).astype(int)
        # Distribute residual budget to largest strata
        remainder = n - alloc.sum()
        for i in np.argsort(-Nh)[:int(remainder)]:
            alloc[i] += 1

        sampled_idx: List[int] = []
        for stratum_idx, n_h in zip(strata, alloc):
            if len(stratum_idx) == 0 or n_h == 0:
                continue
            n_h = min(int(n_h), len(stratum_idx))
            sampled_idx.extend(
                self.rng.choice(stratum_idx, size=n_h, replace=False).tolist()
            )

        if len(sampled_idx) == 0:
            sampled_idx = self.rng.choice(N, size=min(n, N), replace=False).tolist()

        idx = np.asarray(sampled_idx)
        profile = compute_quality_profile(df.iloc[idx], fd_rules)
        profile.n_records = N
        return SamplingResult(profile=profile, n_sampled=len(idx),
                              time_s=time.time() - t0,
                              method="stratified_column")


class StratifiedQualitySampler(BaseSampler):
    """Stratify rows by IQR proxy-score quantile (low / mid / high thirds).

    Neyman optimal allocation: n_h ∝ N_h × std_h(proxy score).
    Uniform draw within strata; uniform HT weights.
    Targets the quality-score distribution rather than column types.
    """

    def __init__(self, n_strata: int = 3, rng=None):
        super().__init__(rng)
        self.n_strata = n_strata

    def sample(self, df: pd.DataFrame, budget_fraction: float = 0.5,
               fd_rules=None) -> SamplingResult:
        t0 = time.time()
        N = len(df)
        n = max(1, int(N * budget_fraction))

        scores = _precompute_error_proxy(df)
        quantiles = np.linspace(0, 100, self.n_strata + 1)
        thresholds = np.percentile(scores, quantiles)

        strata: List[np.ndarray] = []
        for i in range(self.n_strata):
            if i < self.n_strata - 1:
                mask = (scores >= thresholds[i]) & (scores < thresholds[i + 1])
            else:
                mask = scores >= thresholds[i]
            strata.append(np.where(mask)[0])

        Nh = np.array([len(s) for s in strata], dtype=float)
        Sh = np.array([float(scores[s].std()) if len(s) > 1 else 0.0
                       for s in strata])
        weights = Nh * Sh
        total = weights.sum()
        if total == 0:
            weights = Nh.copy()
            total = weights.sum()

        alloc = np.floor(n * weights / total).astype(int)
        remainder = n - alloc.sum()
        order = np.argsort(-Nh)
        for i in range(int(remainder)):
            alloc[order[i % len(order)]] += 1

        sampled_idx: List[int] = []
        for stratum_idx, n_h in zip(strata, alloc):
            if len(stratum_idx) == 0 or n_h == 0:
                continue
            n_h = min(int(n_h), len(stratum_idx))
            sampled_idx.extend(
                self.rng.choice(stratum_idx, size=n_h, replace=False).tolist()
            )

        if len(sampled_idx) == 0:
            sampled_idx = self.rng.choice(N, size=min(n, N), replace=False).tolist()

        idx = np.asarray(sampled_idx)
        profile = compute_quality_profile(df.iloc[idx], fd_rules)
        profile.n_records = N
        return SamplingResult(profile=profile, n_sampled=len(idx),
                              time_s=time.time() - t0,
                              method="stratified_quality")


# ─────────────────────────────────────────────────────────────────────────────
# Cluster sampler
# ─────────────────────────────────────────────────────────────────────────────

class ClusterSampler(BaseSampler):
    """Block-based cluster sampling: divide rows into k consecutive blocks.

    Schema-free, O(N) cost. Natural for ordered data (IoT time series,
    temporally ordered admin records). Selects ceil(b×k) blocks uniformly
    at random and takes all rows from selected blocks, clipping to budget.

    k = max(10, ceil(sqrt(N))) balances cluster granularity vs coverage.
    """

    def sample(self, df: pd.DataFrame, budget_fraction: float = 0.5,
               fd_rules=None) -> SamplingResult:
        t0 = time.time()
        N = len(df)
        n_target = max(1, int(N * budget_fraction))

        k = max(10, int(np.ceil(np.sqrt(N))))
        block_size = int(np.ceil(N / k))
        n_select = max(1, min(int(np.ceil(budget_fraction * k)), k))

        chosen_blocks = self.rng.choice(k, size=n_select, replace=False)
        sampled_idx: List[int] = []
        for c in chosen_blocks:
            start = int(c) * block_size
            end = min(start + block_size, N)
            sampled_idx.extend(range(start, end))

        sampled_idx_arr = np.asarray(sampled_idx)
        if len(sampled_idx_arr) > n_target:
            sampled_idx_arr = self.rng.choice(sampled_idx_arr, size=n_target,
                                              replace=False)

        profile = compute_quality_profile(df.iloc[sampled_idx_arr], fd_rules)
        profile.n_records = N
        return SamplingResult(profile=profile, n_sampled=len(sampled_idx_arr),
                              time_s=time.time() - t0, method="cluster")


# ─────────────────────────────────────────────────────────────────────────────
# Importance sampler
# ─────────────────────────────────────────────────────────────────────────────

class ImportanceSampler(BaseSampler):
    """Direct importance sampling with Horvitz-Thompson correction.

    Draws n rows by weighted reservoir sampling (Efraimidis-Spirakis)
    proportional to the IQR error-proxy score σ_i + ε. Applies HT correction
    w_i ∝ 1/(n × p_i) clipped at the 95th percentile.

    Same proxy as MH/DAG but without a Markov chain — isolates the effect of
    importance-weighted draw from MCMC chain overhead.
    """

    def sample(self, df: pd.DataFrame, budget_fraction: float = 0.5,
               fd_rules=None) -> SamplingResult:
        t0 = time.time()
        N = len(df)
        n = max(1, min(int(N * budget_fraction), N))

        scores = _precompute_error_proxy(df)
        probs = scores + 1e-6
        probs = probs / probs.sum()

        idx = self.rng.choice(N, size=n, replace=False, p=probs)

        # HT weights: w_i ∝ 1 / (N × p_i); clip at 95th percentile
        iw = 1.0 / (probs[idx] * N)
        iw_cap = float(np.percentile(iw, 95))
        if iw_cap > 0:
            iw = np.minimum(iw, iw_cap)
        iw = iw / iw.sum()

        profile = compute_quality_profile(df.iloc[idx], fd_rules,
                                          sample_weights=iw)
        profile.n_records = N
        return SamplingResult(profile=profile, n_sampled=len(idx),
                              time_s=time.time() - t0, method="importance")


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_sampler(method: str, dag: Optional[AttributeDAG] = None,
                rng=None, **kwargs) -> BaseSampler:
    if method == "random_uniform":
        return RandomUniformSampler(rng=rng)
    if method == "geometric":
        return GeometricSampler(rng=rng)
    if method == "yamane":
        return YamaneSampler(rng=rng)
    if method == "metropolis_hastings":
        return MetropolisHastingsSampler(rng=rng, **kwargs)
    if method == "gibbs":
        return GibbsSampler(rng=rng, **kwargs)
    if method == "dag":
        return DAGSampler(dag=dag, rng=rng, **kwargs)
    if method == "dag_uniform":
        return DAGUniformWeightsSampler(dag=dag, rng=rng, **kwargs)
    if method == "stratified_column":
        return StratifiedColumnSampler(rng=rng)
    if method == "stratified_quality":
        return StratifiedQualitySampler(rng=rng)
    if method == "cluster":
        return ClusterSampler(rng=rng)
    if method == "importance":
        return ImportanceSampler(rng=rng)
    if method == "exhaustive":
        raise ValueError("exhaustive is handled directly in ProgressiveProfiler")
    raise ValueError(f"Unknown sampler method: {method!r}")
