"""E7a: Proxy ablation — isolating DAG proposal vs. IQR-based HT weighting.

Compares three conditions on D2 (NYC 311) and D3 (NYPD arrests):
  - dag          : graph-based proposal + IQR-derived HT weights  (full DAG)
  - dag_uniform  : graph-based proposal + uniform HT weights      (proposal only)
  - random_uniform: uniform proposal    + uniform weights          (baseline)

If dag_uniform ≈ random_uniform, the graph proposal is the problem.
If dag_uniform ≈ dag (and both >> random_uniform), the IQR proxy is the problem.
The result disambiguates the root cause of DAG's degradation on real datasets.

Datasets: D2, D3  |  Seeds: 3  |  Budgets: 5%, 10%, 50%
Output  : outputs/E7a/E7a_proxy_ablation.csv
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

OUT_DIR = Path("outputs/E7a")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUT_DIR / "E7a_proxy_ablation.csv"


def _load_done() -> set:
    """Return set of (dataset, method, budget, seed) already in the CSV."""
    if not CSV_PATH.exists():
        return set()
    df = pd.read_csv(CSV_PATH)
    return {
        (r["dataset"], r["method"], r["budget_fraction"], r["seed"])
        for _, r in df.iterrows()
    }


def _append_csv(row: dict) -> None:
    df = pd.DataFrame([row])
    write_header = not CSV_PATH.exists()
    df.to_csv(CSV_PATH, mode="a", header=write_header, index=False)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    t0_total = time.time()

    exp_cfg = cfg.experiment
    datasets: list = list(exp_cfg.get("datasets", ["D2", "D3"]))
    methods: list = list(exp_cfg.get("methods", ["dag", "dag_uniform", "random_uniform"]))
    seeds: list = list(exp_cfg.get("seeds", [42, 123, 456]))
    budgets: list = list(exp_cfg.get("budget_fractions", [0.05, 0.10, 0.50]))

    done = _load_done()
    timing_rows = []

    for ds_name in datasets:
        print(f"\n[E7a] Dataset: {ds_name}")
        df, gt, fd_rules = load_dataset(
            ds_name,
            data_dir=cfg.data.data_dir,
            seed=42,
        )
        if gt is None:
            print(f"  Computing ground truth for {ds_name} ...")
            t0_gt = time.time()
            gt = compute_quality_profile(df, fd_rules)
            gt.n_records = len(df)
            print(f"  GT done in {time.time()-t0_gt:.1f}s: {gt.to_dict()}")

        dag = AttributeDAG.from_correlation(df)

        for budget in budgets:
            for method in methods:
                for seed in seeds:
                    key = (ds_name, method, budget, seed)
                    if key in done:
                        print(f"  SKIP {ds_name}|{method}|{budget}|{seed} (already done)")
                        continue

                    rng = np.random.default_rng(seed)
                    dag_arg = dag if method in ("dag", "dag_uniform") else None
                    profiler = ProgressiveProfiler(
                        method=method, dag=dag_arg,
                        batch_size=cfg.sampler.batch_size, rng=rng,
                    )
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
                    _append_csv(row)
                    done.add(key)

                    timing_rows.append({
                        "dataset": ds_name, "method": method,
                        "budget_fraction": budget, "seed": seed,
                        "time_condition_s": round(t_cond, 3),
                        "time_search_s": row.get("time_search_s", 0),
                        "time_eval_s": row.get("time_eval_s", 0),
                    })

                    err = row.get("rel_err_mean")
                    err_str = f"{err:.4f}" if err is not None else "N/A"
                    print(
                        f"  {ds_name}|{method}|b={budget}|s={seed} "
                        f"err={err_str} {t_cond:.2f}s"
                    )

    timing_df = pd.DataFrame(timing_rows)
    if not timing_df.empty:
        timing_df.to_csv(OUT_DIR / "timing_E7a.csv", index=False)

    t_total = time.time() - t0_total
    print(f"\n{'─'*60}")
    print(f"E7a Timing summary")
    print(f"  Total wall time : {t_total:.1f}s  ({t_total/60:.1f} min)")
    print(f"  Results → {CSV_PATH}")
    print(f"{'─'*60}")

    # Quick summary table
    if CSV_PATH.exists():
        res = pd.read_csv(CSV_PATH)
        if "rel_err_mean" in res.columns:
            summary = (
                res[res["budget_fraction"] == 0.05]
                .groupby(["dataset", "method"])["rel_err_mean"]
                .mean()
                .reset_index()
                .pivot(index="dataset", columns="method", values="rel_err_mean")
            )
            print("\nMean MRE at b=5%:")
            print(summary.to_string())


if __name__ == "__main__":
    main()
