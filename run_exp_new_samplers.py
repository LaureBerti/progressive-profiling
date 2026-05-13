"""Run E1, E3, and E8 for the four new sampling methods only.

New methods: stratified_column, stratified_quality, cluster, importance

Results are appended to the existing E1_results.csv and E3_results.csv so
that the existing method rows are preserved. Run this script once; then the
merged CSVs are ready for analysis.

Usage:
    python run_exp_new_samplers.py                       # all datasets
    python run_exp_new_samplers.py --skip-e3             # E1+E8 only
    python run_exp_new_samplers.py --e8-only             # D7 only
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from profiling import ProgressiveProfiler, AttributeDAG, load_dataset
from profiling.quality_profile import compute_quality_profile

# ── configuration ──────────────────────────────────────────────────────────

NEW_METHODS = [
    "stratified_column",
    "stratified_quality",
    "cluster",
    "importance",
]

E1_DATASETS = ["D1", "D2", "D3", "D4", "D7-synth", "D7-real"]
E3_DATASETS = ["D2", "D3", "D4", "D7-real"]   # budget curves on real datasets

E1_BUDGETS   = [0.05, 0.10, 0.20, 0.30, 0.50]
E3_BUDGETS   = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.0]
SEEDS        = [42, 123, 456]
BATCH_SIZE   = 500   # smaller batch for new methods (no MCMC chain overhead)

DATA_DIR  = "data"
E1_OUT    = Path("outputs/paper_ready/E1/E1_results.csv")
E3_OUT    = Path("outputs/paper_ready/E3/E3_results.csv")

# ── helpers ─────────────────────────────────────────────────────────────────

def _load_and_gt(ds_name: str, data_dir: str = DATA_DIR):
    """Load dataset + ground truth. Returns (df, gt, fd_rules)."""
    df, gt, fd_rules = load_dataset(ds_name, data_dir=data_dir, seed=42)
    if gt is None:
        print(f"  Computing ground truth for {ds_name} exhaustively...")
        t0 = time.time()
        gt = compute_quality_profile(df, fd_rules)
        gt.n_records = len(df)
        print(f"  GT done in {time.time()-t0:.1f}s: {gt.to_dict()}")
    return df, gt, fd_rules


def _append_rows(existing_path: Path, new_rows: list[dict]) -> None:
    """Append new rows to existing CSV, creating it if absent."""
    new_df = pd.DataFrame(new_rows)
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
        merged = pd.concat([existing, new_df], ignore_index=True)
    else:
        merged = new_df
    merged.to_csv(existing_path, index=False)
    print(f"  → {existing_path}  ({len(merged)} total rows, {len(new_df)} new)")


# ── E1: accuracy across all datasets and budgets ───────────────────────────

def run_e1(datasets: list[str], skip_existing: bool = True) -> None:
    print("\n" + "=" * 64)
    print("E1 — new samplers accuracy comparison")
    print("=" * 64)

    # Load existing rows to avoid re-running already-done conditions
    done_keys: set[tuple] = set()
    if skip_existing and E1_OUT.exists():
        existing = pd.read_csv(E1_OUT)
        for _, r in existing.iterrows():
            done_keys.add((r["dataset"], r["method"],
                           float(r["budget_fraction"]), int(r["seed"])))

    all_rows: list[dict] = []
    t0_global = time.time()

    for ds_name in datasets:
        print(f"\n[E1] Dataset: {ds_name}")
        df, gt, fd_rules = _load_and_gt(ds_name)
        dag = AttributeDAG.from_correlation(df)

        for budget in E1_BUDGETS:
            for method in NEW_METHODS:
                for seed in SEEDS:
                    key = (ds_name, method, float(budget), int(seed))
                    if key in done_keys:
                        print(f"  SKIP {ds_name}/{method}/b={budget}/s={seed}")
                        continue

                    rng = np.random.default_rng(seed)
                    profiler = ProgressiveProfiler(
                        method=method, dag=dag,
                        batch_size=BATCH_SIZE, rng=rng,
                    )
                    t0 = time.time()
                    result = profiler.run(
                        df, budget_fraction=budget, ground_truth=gt,
                        dataset_name=ds_name, seed=seed, fd_rules=fd_rules,
                    )
                    elapsed = time.time() - t0

                    row = result.to_row()
                    row["budget_fraction"] = budget
                    all_rows.append(row)

                    err = row.get("rel_err_mean", float("nan"))
                    print(f"  {ds_name} | {method} | b={budget} | s={seed} "
                          f"| err={err:.4f} | {elapsed:.2f}s")

    if all_rows:
        _append_rows(E1_OUT, all_rows)
    else:
        print("  No new rows to append (all conditions already done).")

    print(f"\nE1 total: {time.time()-t0_global:.1f}s")


# ── E3: budget-accuracy trade-off + speedup ────────────────────────────────

def run_e3(datasets: list[str], skip_existing: bool = True) -> None:
    print("\n" + "=" * 64)
    print("E3 — new samplers budget-accuracy trade-off")
    print("=" * 64)

    done_keys: set[tuple] = set()
    if skip_existing and E3_OUT.exists():
        existing = pd.read_csv(E3_OUT)
        for _, r in existing.iterrows():
            done_keys.add((r["dataset"], r["method"],
                           float(r["budget_fraction"]), int(r["seed"])))

    all_rows: list[dict] = []
    t0_global = time.time()

    for ds_name in datasets:
        print(f"\n[E3] Dataset: {ds_name}")
        df, gt, fd_rules = _load_and_gt(ds_name)

        # Measure exhaustive time for speedup computation
        print(f"  Measuring exhaustive time for {ds_name}...")
        t_ex0 = time.time()
        gt_ex = compute_quality_profile(df, fd_rules)
        gt_ex.n_records = len(df)
        t_exhaustive = time.time() - t_ex0
        print(f"  Exhaustive: {t_exhaustive:.2f}s")

        dag = AttributeDAG.from_correlation(df)
        gt_use = gt if gt is not None else gt_ex

        for budget in E3_BUDGETS:
            for method in NEW_METHODS:
                for seed in SEEDS:
                    key = (ds_name, method, float(budget), int(seed))
                    if key in done_keys:
                        continue

                    rng = np.random.default_rng(seed)
                    profiler = ProgressiveProfiler(
                        method=method, dag=dag,
                        batch_size=BATCH_SIZE, rng=rng,
                    )
                    t0 = time.time()
                    result = profiler.run(
                        df, budget_fraction=budget, ground_truth=gt_use,
                        dataset_name=ds_name, seed=seed, fd_rules=fd_rules,
                    )
                    elapsed = time.time() - t0

                    row = result.to_row()
                    row["budget_fraction"] = budget
                    row["t_exhaustive_s"] = round(t_exhaustive, 3)
                    row["speedup"] = round(
                        t_exhaustive / max(result.time_total_s, 1e-6), 2
                    )
                    all_rows.append(row)

                    err = row.get("rel_err_mean", float("nan"))
                    print(f"  {ds_name} | {method} | b={budget} | s={seed} "
                          f"| err={err:.4f} | speedup={row['speedup']}x | {elapsed:.2f}s")

    if all_rows:
        _append_rows(E3_OUT, all_rows)
    else:
        print("  No new rows to append.")

    print(f"\nE3 total: {time.time()-t0_global:.1f}s")


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run new samplers for E1 and E3.")
    parser.add_argument("--skip-e3", action="store_true",
                        help="Run E1 only (skip E3 budget curves).")
    parser.add_argument("--e8-only", action="store_true",
                        help="Run only D7-synth and D7-real (E8 datasets).")
    parser.add_argument("--no-skip", action="store_true",
                        help="Re-run all conditions even if already in CSV.")
    args = parser.parse_args()

    skip_existing = not args.no_skip
    e1_datasets = ["D7-synth", "D7-real"] if args.e8_only else E1_DATASETS
    e3_datasets = ["D7-real"] if args.e8_only else E3_DATASETS

    run_e1(e1_datasets, skip_existing=skip_existing)
    if not args.skip_e3:
        run_e3(e3_datasets, skip_existing=skip_existing)

    print("\nDone. Merged CSVs:")
    print(f"  {E1_OUT}")
    print(f"  {E3_OUT}")


if __name__ == "__main__":
    main()
