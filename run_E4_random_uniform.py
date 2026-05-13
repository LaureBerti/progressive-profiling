"""Run E4 for random_uniform only and append to E4_results.csv."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from profiling import ProgressiveProfiler, AttributeDAG, SyntheticTabularGenerator
from profiling.data_generator import ErrorConfig
from profiling.quality_profile import compute_quality_profile

OUT_DIR = Path("outputs/paper_ready/E4")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXISTING_CSV = OUT_DIR / "E4_results.csv"
TEMP_CSV = OUT_DIR / "E4_results_random_uniform_tmp.csv"

ERROR_TYPE_CONFIGS = {
    "missing":       lambda r: ErrorConfig(missing_rate=r, duplicate_rate=0.01, outlier_rate=0.01, inconsistency_rate=0.01),
    "duplicate":     lambda r: ErrorConfig(missing_rate=0.01, duplicate_rate=r, outlier_rate=0.01, inconsistency_rate=0.01),
    "outlier":       lambda r: ErrorConfig(missing_rate=0.01, duplicate_rate=0.01, outlier_rate=r, inconsistency_rate=0.01),
    "inconsistency": lambda r: ErrorConfig(missing_rate=0.01, duplicate_rate=0.01, outlier_rate=0.01, inconsistency_rate=r),
}

seeds = [42, 123, 456]
error_rates = [0.01, 0.05, 0.10, 0.20, 0.30]
error_types = ["missing", "duplicate", "outlier", "inconsistency"]
n_records = 100_000
budget = 0.5
method = "random_uniform"

rows = []
t0_total = time.time()

for error_type in error_types:
    for rate in error_rates:
        print(f"\n[E4-RU] {error_type} @ {rate*100:.0f}%")
        gen = SyntheticTabularGenerator(rng=np.random.default_rng(0))
        cfg_err = ERROR_TYPE_CONFIGS[error_type](rate)
        dirty, clean, fd_rules = gen.generate(n_records, n_cols=8, config=cfg_err)
        gt = compute_quality_profile(dirty, fd_rules)
        gt.n_records = n_records

        for seed in seeds:
            rng = np.random.default_rng(seed)
            profiler = ProgressiveProfiler(method=method, dag=None, batch_size=500, rng=rng)
            t0 = time.time()
            result = profiler.run(dirty, budget_fraction=budget, ground_truth=gt,
                                  dataset_name="D1", seed=seed, fd_rules=fd_rules)
            t_cond = time.time() - t0
            row = result.to_row()
            row["error_type"] = error_type
            row["error_rate"] = rate
            rows.append(row)
            err_str = f"{row.get('rel_err_mean', 'N/A'):.4f}" if 'rel_err_mean' in row else "N/A"
            print(f"  {error_type} {rate:.0%} | {method} | s={seed} | err={err_str} | {t_cond:.2f}s")

# Save temp CSV
ru_df = pd.DataFrame(rows)
ru_df.to_csv(TEMP_CSV, index=False)
print(f"\nSaved {len(ru_df)} rows to {TEMP_CSV}")

# Merge with existing CSV
if EXISTING_CSV.exists():
    existing_df = pd.read_csv(EXISTING_CSV)
    # Remove any existing random_uniform rows (idempotent)
    existing_df = existing_df[existing_df["method"] != "random_uniform"]
    merged_df = pd.concat([existing_df, ru_df], ignore_index=True)
    merged_df.to_csv(EXISTING_CSV, index=False)
    print(f"Merged into {EXISTING_CSV}: {len(merged_df)} total rows")
else:
    ru_df.to_csv(EXISTING_CSV, index=False)
    print(f"Written to {EXISTING_CSV}")

t_total = time.time() - t0_total
print(f"\nTotal wall time: {t_total:.1f}s ({t_total/60:.1f} min)")

# Summary stats
print("\n--- random_uniform MRE by error_rate ---")
summary = ru_df.groupby("error_rate")["rel_err_mean"].mean()
print(summary.to_string())

print("\n--- random_uniform vs geometric (mean MRE across all conditions) ---")
if EXISTING_CSV.exists():
    full_df = pd.read_csv(EXISTING_CSV)
    cmp = full_df.groupby("method")["rel_err_mean"].mean()
    print(cmp.to_string())
