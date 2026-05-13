"""E4: Error-rate sweep — robustness across injection rates (C-E2).

Sweeps error injection rates 1%–30% for all 4 error types on D1 (synthetic).
Compares dag vs geometric. Expected: dag stable, geometric degrades at ≥20%.
"""
import time
import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).parent))
from profiling import ProgressiveProfiler, AttributeDAG, SyntheticTabularGenerator
from profiling.data_generator import ErrorConfig
from profiling.quality_profile import compute_quality_profile

OUT_DIR = Path("outputs/paper_ready/E4")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ERROR_TYPE_CONFIGS = {
    "missing":       lambda r: ErrorConfig(missing_rate=r, duplicate_rate=0.01, outlier_rate=0.01, inconsistency_rate=0.01),
    "duplicate":     lambda r: ErrorConfig(missing_rate=0.01, duplicate_rate=r, outlier_rate=0.01, inconsistency_rate=0.01),
    "outlier":       lambda r: ErrorConfig(missing_rate=0.01, duplicate_rate=0.01, outlier_rate=r, inconsistency_rate=0.01),
    "inconsistency": lambda r: ErrorConfig(missing_rate=0.01, duplicate_rate=0.01, outlier_rate=0.01, inconsistency_rate=r),
}


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    t0_total = time.time()

    exp_cfg = cfg.experiment
    methods: list = list(exp_cfg.get("methods", ["dag", "geometric"]))
    seeds: list = list(exp_cfg.get("seeds", [42, 123, 456]))
    error_rates: list = list(exp_cfg.get("error_rates", [0.01, 0.05, 0.10, 0.20, 0.30]))
    error_types: list = list(exp_cfg.get("error_types", ["missing", "duplicate", "outlier", "inconsistency"]))
    n_records: int = int(exp_cfg.get("n_records", 100_000))
    budget: float = float(exp_cfg.get("budget_fraction", 0.5))

    rows = []
    timing_rows = []

    for error_type in error_types:
        for rate in error_rates:
            t0_combo = time.time()
            print(f"\n[E4] {error_type} @ {rate*100:.0f}%")
            gen = SyntheticTabularGenerator(rng=np.random.default_rng(0))
            cfg_err = ERROR_TYPE_CONFIGS[error_type](rate)
            dirty, clean, fd_rules = gen.generate(n_records, n_cols=8, config=cfg_err)
            gt = compute_quality_profile(dirty, fd_rules)
            gt.n_records = n_records

            dag = AttributeDAG.from_correlation(dirty)

            for method in methods:
                dag_arg = dag if method == "dag" else None
                for seed in seeds:
                    rng = np.random.default_rng(seed)
                    profiler = ProgressiveProfiler(method=method, dag=dag_arg, batch_size=500, rng=rng)
                    t0_cond = time.time()
                    result = profiler.run(dirty, budget_fraction=budget, ground_truth=gt,
                                         dataset_name="D1", seed=seed, fd_rules=fd_rules)
                    t_cond = time.time() - t0_cond
                    row = result.to_row()
                    row["error_type"] = error_type
                    row["error_rate"] = rate
                    rows.append(row)
                    timing_rows.append({
                        "error_type": error_type, "error_rate": rate,
                        "method": method, "seed": seed,
                        "time_condition_s": round(t_cond, 3),
                        "time_search_s": result.time_search_s,
                        "time_eval_s": result.time_eval_s,
                        "n_candidates": result.n_candidates,
                    })
                    err_str = f"{row.get('rel_err_mean','N/A'):.4f}" if 'rel_err_mean' in row else "N/A"
                    print(f"  {error_type} {rate:.0%} | {method} | s={seed} | err={err_str} | {t_cond:.2f}s")

            print(f"  ↳ {error_type}@{rate:.0%} total: {time.time()-t0_combo:.1f}s")

    results_df = pd.DataFrame(rows)
    timing_df = pd.DataFrame(timing_rows)
    results_df.to_csv(OUT_DIR / "E4_results.csv", index=False)
    timing_df.to_csv(OUT_DIR / "timing_per_condition.csv", index=False)

    t_total = time.time() - t0_total
    print(f"\n{'─'*60}")
    print(f"E4 Timing summary")
    print(f"  Total wall time : {t_total:.1f}s  ({t_total/60:.1f} min)")
    print(f"  Results → {OUT_DIR / 'E4_results.csv'}")
    print(f"{'─'*60}")


if __name__ == "__main__":
    main()
