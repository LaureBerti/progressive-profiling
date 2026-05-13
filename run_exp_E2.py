"""E2: Ablation — DAG-cognizant vs independent MH and Gibbs (C-A2).

Tests whether DAG-guided sampling outperforms independent MCMC when
attribute correlation ρ > 0.3. Controlled via synthetic D1 with injected
correlation at ρ ∈ {0.3, 0.5, 0.7}, and on D_dbp (DBpedia) with fitted DAG.
"""
import time
import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).parent))
from profiling import ProgressiveProfiler, AttributeDAG, load_dataset, SyntheticTabularGenerator
from profiling.data_generator import ErrorConfig

OUT_DIR = Path("outputs/paper_ready/E2")
OUT_DIR.mkdir(parents=True, exist_ok=True)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    t0_total = time.time()

    exp_cfg = cfg.experiment
    methods: list = list(exp_cfg.get("methods", ["dag", "metropolis_hastings", "gibbs"]))
    seeds: list = list(exp_cfg.get("seeds", [42, 123, 456]))
    rho_levels: list = list(exp_cfg.get("correlation_levels", [0.3, 0.5, 0.7]))
    n_records: int = int(exp_cfg.get("n_records", 100_000))
    budget: float = float(exp_cfg.get("budget_fraction", 0.5))

    rows = []
    timing_rows = []

    # ── D1: controlled correlation sweep ──────────────────────────
    for rho in rho_levels:
        t0_rho = time.time()
        print(f"\n[E2] D1 — rho={rho}")
        gen = SyntheticTabularGenerator(rng=np.random.default_rng(0))
        dirty, clean, fd_rules = gen.generate(
            n_records, n_cols=8,
            config=ErrorConfig(
                missing_rate=0.05, duplicate_rate=0.02,
                outlier_rate=0.03, inconsistency_rate=0.02,
                correlation_rho=rho,
            ),
        )
        from profiling.quality_profile import compute_quality_profile
        # GT = exhaustive profile of the dirty dataset (not the clean one)
        gt = compute_quality_profile(dirty, fd_rules)
        gt.n_records = len(dirty)

        # Fit DAG with injected rho (ground-truth dependency structure)
        dag_injected = AttributeDAG.from_correlation(dirty, rho_inject=rho)
        # Independent DAG (uniform — no correlation knowledge) for MH/Gibbs
        dag_uniform = AttributeDAG.from_correlation(dirty, rho_inject=0.0)

        for method in methods:
            dag = dag_injected if method == "dag" else dag_uniform
            for seed in seeds:
                rng = np.random.default_rng(seed)
                profiler = ProgressiveProfiler(method=method, dag=dag, batch_size=500, rng=rng)
                t0 = time.time()
                result = profiler.run(dirty, budget_fraction=budget, ground_truth=gt,
                                      dataset_name="D1", seed=seed, fd_rules=fd_rules)
                t_cond = time.time() - t0
                row = result.to_row()
                row["correlation_rho"] = rho
                row["dataset_variant"] = "D1"
                rows.append(row)
                timing_rows.append({
                    "dataset": "D1", "rho": rho, "method": method, "seed": seed,
                    "time_condition_s": round(t_cond, 3),
                    "time_search_s": result.time_search_s,
                    "time_eval_s": result.time_eval_s,
                    "n_candidates": result.n_candidates,
                })
                err_str = f"{row.get('rel_err_mean','N/A'):.4f}" if 'rel_err_mean' in row else "N/A"
                print(f"  D1 rho={rho} | {method} | s={seed} | err={err_str} | {t_cond:.2f}s")

        print(f"  ↳ D1 rho={rho} total: {time.time()-t0_rho:.1f}s")

    # ── D_dbp: real-world correlation (fitted DAG) ────────────────────
    print(f"\n[E2] D_dbp — real correlation (fitted DAG)")
    t0_d3 = time.time()
    df_d3, gt_d3, fd_d3 = load_dataset("D_dbp", data_dir=cfg.data.data_dir, seed=42)
    if gt_d3 is None:
        from profiling.quality_profile import compute_quality_profile
        gt_d3 = compute_quality_profile(df_d3, fd_d3)
        gt_d3.n_records = len(df_d3)

    dag_d3 = AttributeDAG.from_correlation(df_d3)
    dag_no_dep = AttributeDAG.from_correlation(df_d3, rho_inject=0.0)

    for method in methods:
        dag = dag_d3 if method == "dag" else dag_no_dep
        for seed in seeds:
            rng = np.random.default_rng(seed)
            profiler = ProgressiveProfiler(method=method, dag=dag, batch_size=500, rng=rng)
            t0 = time.time()
            result = profiler.run(df_d3, budget_fraction=budget, ground_truth=gt_d3,
                                  dataset_name="D_dbp", seed=seed, fd_rules=fd_d3)
            t_cond = time.time() - t0
            row = result.to_row()
            row["correlation_rho"] = -1.0  # real (unknown rho)
            row["dataset_variant"] = "D_dbp"
            rows.append(row)
            timing_rows.append({
                "dataset": "D_dbp", "rho": "real", "method": method, "seed": seed,
                "time_condition_s": round(t_cond, 3),
                "time_search_s": result.time_search_s,
                "time_eval_s": result.time_eval_s,
                "n_candidates": result.n_candidates,
            })

    print(f"  ↳ D_dbp total: {time.time()-t0_d3:.1f}s")

    results_df = pd.DataFrame(rows)
    timing_df = pd.DataFrame(timing_rows)
    results_df.to_csv(OUT_DIR / "E2_results.csv", index=False)
    timing_df.to_csv(OUT_DIR / "timing_per_condition.csv", index=False)
    timing_df.to_csv(OUT_DIR / "timing_per_dataset.csv", index=False)

    t_total = time.time() - t0_total
    print(f"\n{'─'*60}")
    print(f"E2 Timing summary")
    print(f"  Total wall time : {t_total:.1f}s  ({t_total/60:.1f} min)")
    print(f"  Results → {OUT_DIR / 'E2_results.csv'}")
    print(f"{'─'*60}")


if __name__ == "__main__":
    main()
