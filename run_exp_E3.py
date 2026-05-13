"""E3: Cost-accuracy trade-off — progressive vs exhaustive profiling (C-E1).

Tests whether our method achieves <5% mean relative error at ≤50% wall-clock
time of exhaustive profiling, on D2 (NYC 311) and D4 (UCI Adult).
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

OUT_DIR = Path("outputs/paper_ready/E3")
OUT_DIR.mkdir(parents=True, exist_ok=True)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    t0_total = time.time()

    exp_cfg = cfg.experiment
    datasets: list = list(exp_cfg.get("datasets", ["D2", "D4"]))
    methods: list = list(exp_cfg.get("methods", ["dag", "exhaustive", "random_uniform"]))
    seeds: list = list(exp_cfg.get("seeds", [42, 123, 456]))
    budgets: list = list(exp_cfg.get("budget_fractions", [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]))

    rows = []
    timing_rows = []

    for ds_name in datasets:
        t0_ds = time.time()
        print(f"\n[E3] Dataset: {ds_name}")

        df, gt, fd_rules = load_dataset(ds_name, data_dir=cfg.data.data_dir, seed=42)

        # Ground truth via exhaustive profiling (also measures exhaustive time)
        print(f"  [E3] Exhaustive profiling (ground truth) for {ds_name}...")
        t0_gt = time.time()
        gt_computed = compute_quality_profile(df, fd_rules)
        gt_computed.n_records = len(df)
        t_exhaustive = time.time() - t0_gt
        print(f"  [E3] Exhaustive: {t_exhaustive:.1f}s → {gt_computed.to_dict()}")

        gt_to_use = gt if gt is not None else gt_computed

        dag = AttributeDAG.from_correlation(df)

        for budget in budgets:
            for method in methods:
                if method == "exhaustive" and budget < 1.0:
                    continue  # exhaustive is always 100%
                for seed in seeds:
                    rng = np.random.default_rng(seed)
                    profiler = ProgressiveProfiler(method=method, dag=dag, batch_size=1000, rng=rng)
                    t0_cond = time.time()
                    result = profiler.run(df, budget_fraction=budget, ground_truth=gt_to_use,
                                         dataset_name=ds_name, seed=seed, fd_rules=fd_rules)
                    t_cond = time.time() - t0_cond

                    row = result.to_row()
                    row["budget_fraction"] = budget
                    row["t_exhaustive_s"] = round(t_exhaustive, 3)
                    row["speedup"] = round(t_exhaustive / max(result.time_total_s, 1e-6), 2)
                    rows.append(row)
                    timing_rows.append({
                        "dataset": ds_name, "method": method, "budget_fraction": budget,
                        "seed": seed, "time_condition_s": round(t_cond, 3),
                        "time_search_s": result.time_search_s,
                        "time_eval_s": result.time_eval_s,
                        "n_candidates": result.n_candidates,
                        "speedup": row["speedup"],
                    })

                    err_str = f"{row.get('rel_err_mean','N/A'):.4f}" if 'rel_err_mean' in row else "N/A"
                    print(
                        f"  {ds_name} | {method} | b={budget} | s={seed} | "
                        f"err={err_str} | speedup={row['speedup']}x | {t_cond:.2f}s"
                    )

        print(f"  ↳ {ds_name} total: {time.time()-t0_ds:.1f}s")

    results_df = pd.DataFrame(rows)
    timing_df = pd.DataFrame(timing_rows)
    results_df.to_csv(OUT_DIR / "E3_results.csv", index=False)
    timing_df.to_csv(OUT_DIR / "timing_per_condition.csv", index=False)
    timing_df.groupby("dataset")["time_condition_s"].sum().reset_index().to_csv(
        OUT_DIR / "timing_per_dataset.csv", index=False
    )

    t_total = time.time() - t0_total
    print(f"\n{'─'*60}")
    print(f"E3 Timing summary")
    print(f"  Total wall time : {t_total:.1f}s  ({t_total/60:.1f} min)")
    print(f"  Results → {OUT_DIR / 'E3_results.csv'}")
    print(f"{'─'*60}")


if __name__ == "__main__":
    main()
