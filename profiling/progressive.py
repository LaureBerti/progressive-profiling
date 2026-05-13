"""ProgressiveProfiler: orchestrates sampling + convergence checking."""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional

from .quality_profile import QualityProfile, compute_quality_profile
from .samplers import BaseSampler, get_sampler, SamplingResult
from .dag import AttributeDAG
from .convergence import ConvergenceChecker


@dataclass
class ProfilerResult:
    method: str
    dataset: str
    n_total: int
    n_sampled: int
    budget_fraction: float
    profile_estimate: QualityProfile
    ground_truth: Optional[QualityProfile]
    relative_errors: Optional[Dict[str, float]]
    converged: bool
    time_total_s: float
    time_search_s: float  # MCMC proposal / index construction
    time_eval_s: float    # quality estimation on accumulated sample
    n_candidates: int     # total records evaluated (for timing manifest)
    acceptance_rate: float
    seed: int

    def to_row(self) -> Dict:
        row = {
            "method": self.method,
            "dataset": self.dataset,
            "n_total": self.n_total,
            "n_sampled": self.n_sampled,
            "budget_fraction": self.budget_fraction,
            "converged": self.converged,
            "time_total_s": round(self.time_total_s, 3),
            "time_search_s": round(self.time_search_s, 3),
            "time_eval_s": round(self.time_eval_s, 3),
            "n_candidates": self.n_candidates,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "seed": self.seed,
        }
        est = self.profile_estimate.to_dict()
        for k, v in est.items():
            row[f"est_{k}"] = round(v, 6)
        if self.relative_errors:
            for k, v in self.relative_errors.items():
                row[f"rel_err_{k}"] = round(v, 6)
        return row


class ProgressiveProfiler:
    """High-level profiler: wraps a sampler, tracks convergence, compares to ground truth."""

    def __init__(
        self,
        method: str,
        dag: Optional[AttributeDAG] = None,
        batch_size: int = 100,
        rng: Optional[np.random.Generator] = None,
    ):
        self.method = method
        self.dag = dag
        self.batch_size = batch_size
        self.rng = rng or np.random.default_rng()
        self._sampler: Optional[BaseSampler] = None

    def _build_sampler(self) -> BaseSampler:
        return get_sampler(self.method, dag=self.dag, rng=self.rng, batch_size=self.batch_size)

    def run(
        self,
        df: pd.DataFrame,
        budget_fraction: float = 0.5,
        ground_truth: Optional[QualityProfile] = None,
        dataset_name: str = "unknown",
        seed: int = 42,
        fd_rules: Optional[Dict[str, str]] = None,
    ) -> ProfilerResult:
        t0_total = time.time()

        if self.method == "exhaustive":
            t0_eval = time.time()
            profile = compute_quality_profile(df, fd_rules)
            profile.n_records = len(df)
            t_eval = time.time() - t0_eval
            return ProfilerResult(
                method="exhaustive",
                dataset=dataset_name,
                n_total=len(df),
                n_sampled=len(df),
                budget_fraction=1.0,
                profile_estimate=profile,
                ground_truth=ground_truth,
                relative_errors=profile.relative_error(ground_truth) if ground_truth else None,
                converged=True,
                time_total_s=time.time() - t0_total,
                time_search_s=0.0,
                time_eval_s=t_eval,
                n_candidates=len(df),
                acceptance_rate=1.0,
                seed=seed,
            )

        sampler = self._build_sampler()

        t0_search = time.time()
        result: SamplingResult = sampler.sample(df, budget_fraction=budget_fraction, fd_rules=fd_rules)
        t_search = time.time() - t0_search

        t0_eval = time.time()
        checker = ConvergenceChecker()
        checker.update(result.profile.to_dict())
        converged = checker.all_converged()
        t_eval = time.time() - t0_eval

        rel_errors = result.profile.relative_error(ground_truth) if ground_truth else None

        return ProfilerResult(
            method=self.method,
            dataset=dataset_name,
            n_total=len(df),
            n_sampled=result.n_sampled,
            budget_fraction=budget_fraction,
            profile_estimate=result.profile,
            ground_truth=ground_truth,
            relative_errors=rel_errors,
            converged=converged,
            time_total_s=time.time() - t0_total,
            time_search_s=t_search,
            time_eval_s=t_eval,
            n_candidates=result.n_sampled,
            acceptance_rate=result.acceptance_rate,
            seed=seed,
        )
