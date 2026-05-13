"""E1: Main comparison — MCMC-guided vs static sampling strategies (C-A1).

Datasets: D1, D2, D3, D4
Methods: dag, metropolis_hastings, random_uniform, geometric, yamane
Seeds: 3 | Budget fractions: [0.1, 0.2, 0.3, 0.5]
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

OUT_DIR = Path("outputs/paper_ready/E1")
OUT_DIR.mkdir(parents=True, exist_ok=True)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    t0_total = time.time()

    exp_cfg = cfg.experiment
    datasets: list = list(exp_cfg.get("datasets", ["D1", "D2", "D3", "D4"]))
    methods: list = list(exp_cfg.get("methods", ["dag", "metropolis_hastings", "random_uniform", "geometric", "yamane"]))
    seeds: list = list(exp_cfg.get("seeds", [42, 123, 456]))
    budgets: list = list(exp_cfg.get("budget_fractions", [0.1, 0.2, 0.3, 0.5]))

    rows = []
    timing_rows = []

    for ds_name in datasets:
        t0_ds = time.time()
        print(f"\n[E1] Dataset: {ds_name}")

        df, gt, fd_rules = load_dataset(
            ds_name,
            data_dir=cfg.data.data_dir,
            n_records=cfg.data.get("n_records_D1") if ds_name == "D1" else None,
            seed=42,
            error_rate=cfg.data.get("error_rate", 0.05),
        )

        # Compute ground truth exhaustively if not provided (real datasets)
        if gt is None:
            print(f"  [E1] Computing ground truth for {ds_name} exhaustively...")
            t0_gt = time.time()
            gt = compute_quality_profile(df, fd_rules)
            gt.n_records = len(df)
            print(f"  [E1] GT computed in {time.time()-t0_gt:.1f}s: {gt.to_dict()}")

        # Fit DAG once per dataset
        dag = AttributeDAG.from_correlation(df)

        for budget in budgets:
            for method in methods:
                for seed in seeds:
                    rng = np.random.default_rng(seed)
                    profiler = ProgressiveProfiler(method=method, dag=dag, batch_size=cfg.sampler.batch_size, rng=rng)

                    t0_cond = time.time()
                    result = profiler.run(
                        df,
                        budget_fraction=budget,
                        ground_truth=gt,
                        dataset_name=ds_name,
                        seed=seed,
                        fd_rules=fd_rules,
                    )
                    t_cond = time.time() - t0_cond

                    row = result.to_row()
                    row["budget_fraction"] = budget
                    rows.append(row)

                    timing_rows.append({
                        "dataset": ds_name,
                        "method": method,
                        "budget_fraction": budget,
                        "seed": seed,
                        "time_condition_s": round(t_cond, 3),
                        "time_search_s": row.get("time_search_s", 0),
                        "time_eval_s": row.get("time_eval_s", 0),
                        "n_candidates": row.get("n_candidates", 0),
                    })

                    err_str = f"{row.get('rel_err_mean', 'N/A'):.4f}" if 'rel_err_mean' in row else "N/A"
                    print(
                        f"  {ds_name} | {method} | budget={budget} | seed={seed} | "
                        f"rel_err={err_str} | search={result.time_search_s:.2f}s "
                        f"eval={result.time_eval_s:.2f}s total={t_cond:.2f}s"
                    )

        ds_elapsed = time.time() - t0_ds
        print(f"  ↳ {ds_name} total: {ds_elapsed:.1f}s")

    results_df = pd.DataFrame(rows)
    timing_df = pd.DataFrame(timing_rows)

    results_df.to_csv(OUT_DIR / "E1_results.csv", index=False)
    timing_df.to_csv(OUT_DIR / "timing_per_condition.csv", index=False)
    timing_df.groupby("dataset")["time_condition_s"].sum().reset_index().to_csv(
        OUT_DIR / "timing_per_dataset.csv", index=False
    )

    t_total = time.time() - t0_total
    mean_cond = timing_df["time_condition_s"].mean() if not timing_df.empty else 0
    slowest = timing_df.loc[timing_df["time_condition_s"].idxmax()] if not timing_df.empty else None

    print(f"\n{'─'*60}")
    print(f"E1 Timing summary")
    print(f"  Total wall time : {t_total:.1f}s  ({t_total/60:.1f} min)")
    print(f"  Mean per condition: {mean_cond:.1f}s")
    if slowest is not None:
        print(f"  Slowest condition : {slowest['dataset']}/{slowest['method']} ({slowest['time_condition_s']:.1f}s)")
    print(f"  Results → {OUT_DIR / 'E1_results.csv'}")
    print(f"{'─'*60}")


if __name__ == "__main__":
    main()
