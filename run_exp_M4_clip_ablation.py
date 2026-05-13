"""M4: HT clipping sensitivity ablation on D2 (NYC 311).

Tests five clip-percentile thresholds for Horvitz-Thompson weights applied to
the DAG sampler: None (no clipping), 80th, 90th, 95th (baseline), 99th.

Produces: outputs/paper_ready/M4/M4_clip_ablation.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from profiling import load_dataset, AttributeDAG
from profiling.quality_profile import compute_quality_profile
from profiling.samplers import _precompute_error_proxy, DAGSampler

OUT_DIR = Path("outputs/paper_ready/M4")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456]
BUDGETS = [0.05, 0.1, 0.2, 0.3, 0.5]
CLIP_PERCENTILES = [None, 80, 90, 95, 99]
BATCH_SIZE = 200


def _run_dag_with_clip(df, dag, budget_fraction, seed, clip_pct, fd_rules):
    """Replicate DAGSampler.sample() with a custom HT clip percentile."""
    rng = np.random.default_rng(seed)
    N = len(df)
    budget = max(BATCH_SIZE, int(N * budget_fraction))

    # Borrow DAGSampler for proposals/acceptance but run the loop ourselves
    sampler = DAGSampler(dag=dag, batch_size=BATCH_SIZE, rng=rng)
    scores = _precompute_error_proxy(df)
    sampler._score_thresh = float(np.median(scores))
    sampler._scores_uniform = bool(float(scores.max()) == float(scores.min()))

    # Warm start biased toward high-score rows
    init_n = min(BATCH_SIZE, N)
    probs = scores + 1e-6
    probs /= probs.sum()
    init_idx = rng.choice(N, size=init_n, replace=False, p=probs)

    sampled = np.zeros(N, dtype=bool)
    sampled[init_idx] = True
    n_sampled = int(init_n)
    curr_score = float(scores[init_idx].mean())

    pool = np.setdiff1d(np.arange(N, dtype=np.int64), init_idx)
    pool_size = len(pool)
    pool_pos = np.full(N, N, dtype=np.int64)
    pool_pos[pool] = np.arange(pool_size, dtype=np.int64)

    while n_sampled < budget:
        if pool_size == 0:
            break
        unsampled_idx = pool[:pool_size]
        proposed_idx = sampler._propose(unsampled_idx, scores, df)
        prop_score = float(scores[proposed_idx].mean())
        if sampler._accept(curr_score, prop_score):
            add_idx = proposed_idx
        else:
            force_n = max(1, min(BATCH_SIZE // 4, pool_size))
            add_idx = rng.choice(pool[:pool_size], size=force_n, replace=False)

        positions = pool_pos[add_idx]
        order = np.argsort(positions)[::-1]
        for k in order:
            idx = int(add_idx[k])
            p = int(pool_pos[idx])
            if p >= pool_size:
                continue
            pool_size -= 1
            if p < pool_size:
                last = int(pool[pool_size])
                pool[p] = last
                pool_pos[last] = p
            pool_pos[idx] = N

        n_new = len(add_idx)
        curr_score = (curr_score * n_sampled + float(scores[add_idx].sum())) / (n_sampled + n_new)
        n_sampled += n_new
        sampled[add_idx] = True

    final_idx = np.where(sampled)[0][:budget]
    final_scores = scores[final_idx]
    iw = 1.0 / (final_scores + 1e-6)

    # Apply clip at the requested percentile (None = no clipping)
    if clip_pct is not None:
        iw_cap = float(np.percentile(iw, clip_pct))
        if iw_cap > 0:
            iw = np.minimum(iw, iw_cap)
    iw = iw / iw.sum()

    profile = compute_quality_profile(df.iloc[final_idx], fd_rules, sample_weights=iw)
    profile.n_records = N
    return profile


def _rel_err(profile, gt):
    vals = [
        (profile.missing_rate, gt.missing_rate),
        (profile.duplicate_rate, gt.duplicate_rate),
        (profile.outlier_rate, gt.outlier_rate),
        (profile.inconsistency_rate, gt.inconsistency_rate),
    ]
    return sum(min(abs(e - g) / max(g, 0.05), 1.0) for e, g in vals) / 4


def main():
    t0_total = time.time()
    print("[M4] Loading D2 (NYC 311)...")
    df, gt, fd_rules = load_dataset("D2", data_dir="data", seed=42)
    if gt is None:
        print("[M4] Computing ground truth exhaustively...")
        gt = compute_quality_profile(df, fd_rules)
        gt.n_records = len(df)
    dag = AttributeDAG.from_correlation(df)
    print(f"[M4] D2: {len(df):,} rows. GT={gt.to_dict()}")

    rows = []
    for clip_pct in CLIP_PERCENTILES:
        label = "none" if clip_pct is None else str(clip_pct)
        for budget in BUDGETS:
            for seed in SEEDS:
                t0 = time.time()
                profile = _run_dag_with_clip(df, dag, budget, seed, clip_pct, fd_rules)
                err = _rel_err(profile, gt)
                elapsed = time.time() - t0
                rows.append({
                    "clip_percentile": label,
                    "budget_fraction": budget,
                    "seed": seed,
                    "rel_err_mean": round(err, 4),
                    "time_condition_s": round(elapsed, 3),
                })
                print(f"  clip={label:4s} | b={budget} | seed={seed} | "
                      f"rel_err={err:.4f} | {elapsed:.2f}s")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_DIR / "M4_clip_ablation.csv", index=False)

    summary = (
        df_out.groupby("clip_percentile")["rel_err_mean"]
        .mean()
        .round(4)
        .reindex(["none", "80", "90", "95", "99"])
    )
    print(f"\n{'─'*55}")
    print("M4 clip ablation — mean rel_err per threshold (dag, D2)")
    print(summary.to_string())
    print(f"\nTotal: {time.time()-t0_total:.1f}s → {OUT_DIR / 'M4_clip_ablation.csv'}")
    print(f"{'─'*55}")


if __name__ == "__main__":
    main()
