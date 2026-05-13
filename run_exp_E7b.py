"""E7b: MCMC hyperparameter sweep — batch size B ∈ {100, 200, 500, 1000}.

Tests whether DAG's reported results (B=500, λ=2.0) reflect a deliberate
sweet spot or an arbitrary default. Runs DAG and MH across four batch sizes
at b=5% on D2 (NYC 311, the primary real dataset used in E3).

If DAG is robust across B values, the paper's parameter choice is defensible.
If a smaller B sharply improves accuracy or speed, the reported numbers are
sensitive to the choice — a reviewer weakness.

Dataset: D2  |  Budget: 5%  |  Seeds: 3  |  B: {100, 200, 500, 1000}
Output : outputs/E7b/E7b_sweep.csv
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

OUT_DIR = Path("outputs/E7b")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUT_DIR / "E7b_sweep.csv"


def _load_done() -> set:
    if not CSV_PATH.exists():
        return set()
    df = pd.read_csv(CSV_PATH)
    return {
        (r["dataset"], r["method"], r["batch_size"], r["seed"])
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
    datasets: list = list(exp_cfg.get("datasets", ["D2"]))
    methods: list = list(exp_cfg.get("methods", ["dag", "metropolis_hastings"]))
    seeds: list = list(exp_cfg.get("seeds", [42, 123, 456]))
    batch_sizes: list = list(exp_cfg.get("batch_sizes", [100, 200, 500, 1000]))
    budget: float = float(exp_cfg.get("budget_fraction", 0.05))

    done = _load_done()
    timing_rows = []

    for ds_name in datasets:
        print(f"\n[E7b] Dataset: {ds_name}")
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

        for B in batch_sizes:
            for method in methods:
                dag_arg = dag if method == "dag" else None
                for seed in seeds:
                    key = (ds_name, method, B, seed)
                    if key in done:
                        print(f"  SKIP {ds_name}|{method}|B={B}|s={seed} (already done)")
                        continue

                    rng = np.random.default_rng(seed)
                    profiler = ProgressiveProfiler(
                        method=method, dag=dag_arg,
                        batch_size=B, rng=rng,
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
                    row["batch_size"] = B
                    _append_csv(row)
                    done.add(key)

                    timing_rows.append({
                        "dataset": ds_name, "method": method,
                        "batch_size": B, "seed": seed,
                        "time_condition_s": round(t_cond, 3),
                        "time_search_s": row.get("time_search_s", 0),
                        "time_eval_s": row.get("time_eval_s", 0),
                    })

                    err = row.get("rel_err_mean")
                    err_str = f"{err:.4f}" if err is not None else "N/A"
                    print(
                        f"  {ds_name}|{method}|B={B}|s={seed} "
                        f"err={err_str} {t_cond:.2f}s"
                    )

    timing_df = pd.DataFrame(timing_rows)
    if not timing_df.empty:
        timing_df.to_csv(OUT_DIR / "timing_E7b.csv", index=False)

    t_total = time.time() - t0_total
    print(f"\n{'─'*60}")
    print(f"E7b Timing summary")
    print(f"  Total wall time : {t_total:.1f}s  ({t_total/60:.1f} min)")
    print(f"  Results → {CSV_PATH}")
    print(f"{'─'*60}")

    # Quick summary table
    if CSV_PATH.exists():
        res = pd.read_csv(CSV_PATH)
        if "rel_err_mean" in res.columns and "batch_size" in res.columns:
            summary = (
                res.groupby(["method", "batch_size"])["rel_err_mean"]
                .mean()
                .reset_index()
                .pivot(index="batch_size", columns="method", values="rel_err_mean")
            )
            print(f"\nMean MRE at b=5% by batch size (D2):")
            print(summary.to_string())


if __name__ == "__main__":
    main()
