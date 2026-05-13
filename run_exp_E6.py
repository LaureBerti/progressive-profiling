"""E6: XXL Real-Dataset Scalability — budget sweep on D6 (7M+ real rows).

Two datasets, run one at a time:

  D6-A  TWO CENTURIES OF UM RACES  (~7.4M rows, 11 cols)
        Local file: data/ultra-marathon.zip
        Issues: ~12% missing (Country, Age), duplicate athlete records,
                outliers in finish times (DNS/DNF entries)

  D6-B  NYC Yellow Taxi Jan–Feb 2023  (~7.1M rows, 19 cols)
        Local files: data/yellow_tripdata_2023-01.parquet
                     data/yellow_tripdata_2023-02.parquet
        Issues: mostly-null columns (ehail_fee), fare/distance outliers,
                cash trips with tip_amount > 0 (FD violation)

Usage:
    python run_exp_E6.py experiment.dataset=ultra_marathon
    python run_exp_E6.py experiment.dataset=nyc_taxi

    # Quick smoke test (100K rows):
    python run_exp_E6.py experiment.dataset=ultra_marathon experiment.n_records=100000

    # Reduced seed set:
    python run_exp_E6.py experiment.dataset=nyc_taxi experiment.seeds=[42,123,456]

Outputs (separate per dataset):
    outputs/E6/E6_results_ultra_marathon.csv
    outputs/E6/E6_results_nyc_taxi.csv
    outputs/E6/timing_ultra_marathon.csv
    outputs/E6/timing_nyc_taxi.csv
"""
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).parent))
from profiling import ProgressiveProfiler, AttributeDAG, load_dataset
from profiling.quality_profile import compute_quality_profile, QualityProfile

OUT_DIR = Path("outputs/E6")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_CHOICES = ("ultra_marathon", "nyc_taxi")

# ─────────────────────────────────────────────────────────────────────────────
# Dataset loaders  (local files only — no network calls)
# ─────────────────────────────────────────────────────────────────────────────

def _load_ultra_marathon(
    data_dir: Path,
    n_records: Optional[int] = None,
    seed: int = 42,
) -> Tuple[pd.DataFrame, QualityProfile, Dict]:
    """Load D6-A from data/ultra-marathon.zip (local file required)."""
    zip_path = data_dir / "ultra-marathon.zip"
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {zip_path}\n"
            "Place ultra-marathon.zip (from Kaggle) in the data/ directory."
        )

    cache_parquet = data_dir / "D6_umraces.parquet"
    gt_path = data_dir / "D6_umraces.gt.json"

    # Parse zip → parquet cache on first run
    if not cache_parquet.exists():
        print(f"  Extracting {zip_path} ...")
        t0 = time.time()
        import zipfile
        with zipfile.ZipFile(zip_path) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                raise RuntimeError(f"No CSV found inside {zip_path}. Contents: {zf.namelist()}")
            csv_name = csv_names[0]
            print(f"  Reading {csv_name} ...")
            with zf.open(csv_name) as fh:
                df = pd.read_csv(fh, encoding="utf-8", low_memory=False, dtype=str)
        print(f"  Parsed in {time.time()-t0:.1f}s  → {len(df):,} rows × {len(df.columns)} cols")

        # Cast numeric columns
        for col in df.columns:
            if any(k in col.lower() for k in ("year", "age", "time", "speed", "distance", "perf")):
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df.to_parquet(cache_parquet, index=False)
        print(f"  Cached → {cache_parquet}")
    else:
        t0 = time.time()
        df = pd.read_parquet(cache_parquet)
        print(f"  Loaded from cache in {time.time()-t0:.1f}s  → {len(df):,} rows")

    if n_records and len(df) > n_records:
        df = df.sample(n=n_records, random_state=seed).reset_index(drop=True)
        print(f"  Subsampled to {len(df):,} rows")

    gt = _load_or_compute_gt(df, gt_path, fd_rules={})
    return df, gt, {}


def _load_nyc_taxi(
    data_dir: Path,
    n_records: Optional[int] = None,
    seed: int = 42,
) -> Tuple[pd.DataFrame, QualityProfile, Dict]:
    """Load D6-B from data/yellow_tripdata_2023-0[12].parquet (local files required)."""
    jan = data_dir / "yellow_tripdata_2023-01.parquet"
    feb = data_dir / "yellow_tripdata_2023-02.parquet"
    missing = [str(p) for p in (jan, feb) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Dataset file(s) not found:\n  " + "\n  ".join(missing) + "\n"
            "Place the Parquet files in the data/ directory."
        )

    cache_parquet = data_dir / "D6_nyctaxi_2023q1.parquet"
    gt_path = data_dir / "D6_nyctaxi_2023q1.gt.json"

    if not cache_parquet.exists():
        print("  Combining Jan + Feb parquet files ...")
        t0 = time.time()
        frames = [pd.read_parquet(p) for p in (jan, feb)]
        df = pd.concat(frames, ignore_index=True)
        print(f"  Combined in {time.time()-t0:.1f}s  → {len(df):,} rows × {len(df.columns)} cols")
        df.to_parquet(cache_parquet, index=False)
        print(f"  Cached → {cache_parquet}")
    else:
        t0 = time.time()
        df = pd.read_parquet(cache_parquet)
        print(f"  Loaded from cache in {time.time()-t0:.1f}s  → {len(df):,} rows")

    if n_records and len(df) > n_records:
        df = df.sample(n=n_records, random_state=seed).reset_index(drop=True)
        print(f"  Subsampled to {len(df):,} rows")

    # Known FD: VendorID → RatecodeID (not always, but a useful probe)
    fd_rules = {"VendorID": "RatecodeID"}
    gt = _load_or_compute_gt(df, gt_path, fd_rules=fd_rules)
    return df, gt, fd_rules


def _load_or_compute_gt(
    df: pd.DataFrame,
    gt_path: Path,
    fd_rules: Dict,
) -> QualityProfile:
    """Load ground-truth profile from cache or compute exhaustively and cache."""
    if gt_path.exists():
        with open(gt_path) as f:
            d = json.load(f)
        gt = QualityProfile(
            missing_rate=d["missing_rate"],
            duplicate_rate=d["duplicate_rate"],
            outlier_rate=d["outlier_rate"],
            inconsistency_rate=d["inconsistency_rate"],
            n_records=len(df),
            sample_size=len(df),
        )
        print(f"  GT loaded from cache: MV={gt.missing_rate:.4f}  dup={gt.duplicate_rate:.4f}  "
              f"out={gt.outlier_rate:.4f}  inc={gt.inconsistency_rate:.4f}")
        return gt

    print(f"  Computing GT exhaustively on {len(df):,} rows ...")
    t0 = time.time()
    gt = compute_quality_profile(df, fd_rules=fd_rules)
    gt.n_records = len(df)
    elapsed = time.time() - t0
    print(f"  GT computed in {elapsed:.1f}s: MV={gt.missing_rate:.4f}  "
          f"dup={gt.duplicate_rate:.4f}  out={gt.outlier_rate:.4f}  "
          f"inc={gt.inconsistency_rate:.4f}")
    with open(gt_path, "w") as fh:
        json.dump({
            "missing_rate": gt.missing_rate,
            "duplicate_rate": gt.duplicate_rate,
            "outlier_rate": gt.outlier_rate,
            "inconsistency_rate": gt.inconsistency_rate,
        }, fh, indent=2)
    print(f"  GT cached → {gt_path}")
    return gt


# ─────────────────────────────────────────────────────────────────────────────
# Resume helper
# ─────────────────────────────────────────────────────────────────────────────

def _load_completed_keys(results_file: Path) -> set:
    if not results_file.exists():
        return set()
    try:
        df_done = pd.read_csv(results_file)
        if df_done.empty or "method" not in df_done.columns:
            return set()
        return {f"{r['method']}|{r['budget_fraction']}|{r['seed']}"
                for _, r in df_done.iterrows()}
    except Exception:
        return set()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

@hydra.main(config_path="conf", config_name="config_e6", version_base=None)
def main(cfg: DictConfig) -> None:
    t0_total = time.time()
    exp_cfg = cfg.experiment

    # -- Parameters -----------------------------------------------------------
    dataset: str = str(exp_cfg.get("dataset", "ultra_marathon"))
    if dataset not in DATASET_CHOICES:
        raise ValueError(
            f"experiment.dataset must be one of {DATASET_CHOICES}, got '{dataset}'.\n"
            "Example: python run_exp_E6.py experiment.dataset=nyc_taxi"
        )

    methods: List[str] = list(exp_cfg.get("methods", ["dag", "random_uniform", "geometric"]))
    seeds: List[int] = list(exp_cfg.get("seeds", [42, 123, 456, 789, 1011, 1314, 1617, 1920, 2223, 2526]))
    budgets: List[float] = list(exp_cfg.get("budgets", [0.01, 0.02, 0.05, 0.10, 0.20]))
    n_records: Optional[int] = exp_cfg.get("n_records", None)
    run_exhaustive_gt: bool = bool(exp_cfg.get("run_exhaustive_gt", True))
    data_dir = Path(cfg.data.data_dir)

    # Per-dataset output files so runs don't overwrite each other
    results_file = OUT_DIR / f"E6_results_{dataset}.csv"
    timing_file  = OUT_DIR / f"timing_{dataset}.csv"

    print(f"\n{'═'*62}")
    print(f"E6 — XXL Real-Dataset Scalability")
    print(f"  Dataset : {dataset}")
    print(f"  Methods : {methods}")
    print(f"  Budgets : {budgets}")
    print(f"  Seeds   : {seeds}  (n={len(seeds)})")
    print(f"  n_records cap : {n_records or 'full'}")
    print(f"  Results file  : {results_file}")
    print(f"{'═'*62}\n")

    # -- Load dataset ---------------------------------------------------------
    print(f"[E6] Loading dataset '{dataset}' ...")
    t0_load = time.time()

    if dataset == "ultra_marathon":
        df, gt, fd_rules = _load_ultra_marathon(data_dir, n_records=n_records, seed=seeds[0])
        dataset_label = "D6_umraces"
    else:  # nyc_taxi
        df, gt, fd_rules = _load_nyc_taxi(data_dir, n_records=n_records, seed=seeds[0])
        dataset_label = "D6_nyctaxi"

    n_total = len(df)
    print(f"\n[E6] Ready: {n_total:,} rows × {len(df.columns)} cols  "
          f"(loaded in {time.time()-t0_load:.1f}s)")
    print(f"[E6] GT: MV={gt.missing_rate:.4f}  dup={gt.duplicate_rate:.4f}  "
          f"out={gt.outlier_rate:.4f}  inc={gt.inconsistency_rate:.4f}")

    # -- Build AttributeDAG (once, shared across seeds) -----------------------
    print("[E6] Building AttributeDAG ...")
    t0_dag = time.time()
    dag = AttributeDAG.from_correlation(df)
    print(f"  DAG built in {time.time()-t0_dag:.1f}s")

    # -- Resume logic ---------------------------------------------------------
    completed_keys = _load_completed_keys(results_file)
    if completed_keys:
        print(f"[E6] Resume: {len(completed_keys)} conditions already done, skipping.")
    write_header = not results_file.exists() or results_file.stat().st_size == 0

    rows: List[Dict] = []
    timing_rows: List[Dict] = []
    total_conditions = len(methods) * len(budgets) * len(seeds)
    done_count = len(completed_keys)

    # -- Experiment loop ------------------------------------------------------
    for budget in budgets:
        t0_budget = time.time()
        print(f"\n[E6] Budget = {budget*100:.0f}%  "
              f"(sampling {int(n_total * budget):,} of {n_total:,} rows)")

        for method in methods:
            t0_method = time.time()

            for seed in seeds:
                key = f"{method}|{budget}|{seed}"
                if key in completed_keys:
                    continue

                rng = np.random.default_rng(seed)
                profiler = ProgressiveProfiler(
                    method=method, dag=dag, batch_size=1000, rng=rng
                )

                t0_cond = time.time()
                result = profiler.run(
                    df,
                    budget_fraction=budget,
                    ground_truth=gt,
                    dataset_name=dataset_label,
                    seed=seed,
                    fd_rules=fd_rules,
                )
                t_cond = time.time() - t0_cond

                row = result.to_row()
                row["budget_fraction"] = budget
                row["n_total"] = n_total
                row["dataset"] = dataset
                rows.append(row)
                timing_rows.append({
                    "method": method,
                    "budget_fraction": budget,
                    "seed": seed,
                    "time_condition_s": round(t_cond, 3),
                    "time_search_s": result.time_search_s,
                    "time_eval_s": result.time_eval_s,
                    "n_candidates": result.n_candidates,
                })

                done_count += 1
                pct = done_count / total_conditions * 100
                err = row.get("rel_err_mean", "N/A")
                err_str = f"{err:.4f}" if isinstance(err, float) else err
                print(
                    f"  {dataset_label} | b={budget*100:.0f}% | {method:16s} | "
                    f"s={seed:5d} | MRE={err_str} | "
                    f"search={result.time_search_s:.2f}s  "
                    f"eval={result.time_eval_s:.2f}s  "
                    f"total={t_cond:.2f}s  [{pct:.0f}%]"
                )

                # Incremental write — safe against crashes
                new_df = pd.DataFrame([row])
                new_df.to_csv(
                    results_file, index=False,
                    mode="w" if write_header else "a",
                    header=write_header,
                )
                write_header = False

            print(f"  ↳ b={budget*100:.0f}% | {method}: {time.time()-t0_method:.1f}s")
        print(f"  ↳ Budget {budget*100:.0f}% done: {time.time()-t0_budget:.1f}s")

    # -- Exhaustive reference run ---------------------------------------------
    if run_exhaustive_gt:
        print(f"\n[E6] Exhaustive reference run (full dataset, seed=42) ...")
        t0_ex = time.time()
        profiler_ex = ProgressiveProfiler(
            method="exhaustive", dag=dag, batch_size=1000,
            rng=np.random.default_rng(42),
        )
        result_ex = profiler_ex.run(
            df, budget_fraction=1.0, ground_truth=gt,
            dataset_name=dataset_label, seed=42, fd_rules=fd_rules,
        )
        t_ex = time.time() - t0_ex
        row_ex = result_ex.to_row()
        row_ex.update({"budget_fraction": 1.0, "n_total": n_total, "dataset": dataset})
        pd.DataFrame([row_ex]).to_csv(results_file, index=False, mode="a", header=False)
        timing_rows.append({
            "method": "exhaustive", "budget_fraction": 1.0, "seed": 42,
            "time_condition_s": round(t_ex, 3),
            "time_search_s": 0.0,
            "time_eval_s": result_ex.time_eval_s,
            "n_candidates": n_total,
        })
        print(f"  Exhaustive: MRE={row_ex.get('rel_err_mean','N/A')} | {t_ex:.1f}s")

    # -- Save timing ----------------------------------------------------------
    timing_df = pd.DataFrame(timing_rows)
    timing_df.to_csv(timing_file, index=False)

    # -- Summary banner -------------------------------------------------------
    t_total = time.time() - t0_total
    mean_cond = timing_df["time_condition_s"].mean() if not timing_df.empty else 0.0
    print(f"\n{'─'*62}")
    print(f"E6 ({dataset}) — done")
    print(f"  Total wall time   : {t_total:.1f}s  ({t_total/60:.1f} min)")
    print(f"  Mean per condition: {mean_cond:.1f}s")
    print(f"  Results  → {results_file}")
    print(f"  Timing   → {timing_file}")
    print(f"{'─'*62}")


if __name__ == "__main__":
    main()
