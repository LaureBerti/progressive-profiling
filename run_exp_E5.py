"""E5: Scalability — profiling cost and accuracy as N scales (C-E5).

Uses D5: Outsourcing pre-generated Adult-derived tabular data with injected errors.
N ∈ {10^4, 10^5, 5×10^5, 10^6, 5×10^6}. Note: N=5M uses seed=42 only (single file).
Compares: dag, random_uniform, exhaustive.
"""
import time
import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).parent))
from profiling import ProgressiveProfiler, AttributeDAG, load_dataset
from profiling.quality_profile import compute_quality_profile

OUT_DIR = Path("outputs/paper_ready/E5")
OUT_DIR.mkdir(parents=True, exist_ok=True)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    t0_total = time.time()

    exp_cfg = cfg.experiment
    methods: list = list(exp_cfg.get("methods", ["dag", "random_uniform", "exhaustive"]))
    seeds: list = list(exp_cfg.get("seeds", [42, 123, 456]))
    scale_sizes: list = list(exp_cfg.get("scale_sizes", [10_000, 100_000, 500_000, 1_000_000, 5_000_000]))
    budget: float = float(exp_cfg.get("budget_fraction", 0.5))
    error_rate: float = float(exp_cfg.get("error_rate", 0.05))
    # N=5M: only seed=42 has a pre-generated Outsourcing file
    _seeds_for_n = {5_000_000: [42]}

    rows = []
    timing_rows = []

    for n_scale in scale_sizes:
        t0_n = time.time()
        active_seeds = _seeds_for_n.get(n_scale, seeds)
        print(f"\n[E5] N={n_scale:,} | seeds={active_seeds}")

        df, gt, fd_rules = load_dataset(
            "D5",
            data_dir=cfg.data.data_dir,
            n_records=n_scale,
            seed=42,
            error_rate=error_rate,
        )

        if gt is None:
            t0_gt = time.time()
            gt = compute_quality_profile(df, fd_rules)
            gt.n_records = len(df)
            print(f"  GT computed in {time.time()-t0_gt:.1f}s")

        dag = AttributeDAG.from_correlation(df)

        for method in methods:
            for seed in active_seeds:
                rng = np.random.default_rng(seed)
                profiler = ProgressiveProfiler(method=method, dag=dag, batch_size=1000, rng=rng)

                t0_cond = time.time()
                result = profiler.run(df, budget_fraction=budget, ground_truth=gt,
                                      dataset_name=f"D5_n{n_scale}", seed=seed, fd_rules=fd_rules)
                t_cond = time.time() - t0_cond

                row = result.to_row()
                row["n_scale"] = n_scale
                rows.append(row)
                timing_rows.append({
                    "n_scale": n_scale, "method": method, "seed": seed,
                    "time_condition_s": round(t_cond, 3),
                    "time_search_s": result.time_search_s,
                    "time_eval_s": result.time_eval_s,
                    "n_candidates": result.n_candidates,
                })

                err_str = f"{row.get('rel_err_mean','N/A'):.4f}" if 'rel_err_mean' in row else "N/A"
                print(f"  N={n_scale:,} | {method} | s={seed} | err={err_str} | {t_cond:.2f}s")

        print(f"  ↳ N={n_scale:,} total: {time.time()-t0_n:.1f}s")

    results_df = pd.DataFrame(rows)
    timing_df = pd.DataFrame(timing_rows)
    results_df.to_csv(OUT_DIR / "E5_results.csv", index=False)
    timing_df.to_csv(OUT_DIR / "timing_per_condition.csv", index=False)
    timing_df.groupby("n_scale")["time_condition_s"].sum().reset_index().to_csv(
        OUT_DIR / "timing_per_dataset.csv", index=False
    )

    t_total = time.time() - t0_total
    mean_cond = timing_df["time_condition_s"].mean() if not timing_df.empty else 0
    print(f"\n{'─'*60}")
    print(f"E5 Timing summary")
    print(f"  Total wall time : {t_total:.1f}s  ({t_total/60:.1f} min)")
    print(f"  Mean per condition: {mean_cond:.1f}s")
    print(f"  Results → {OUT_DIR / 'E5_results.csv'}")
    print(f"{'─'*60}")


if __name__ == "__main__":
    main()
