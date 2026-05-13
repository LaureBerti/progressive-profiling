# Data Quality Profiling at Scale with Progressive Sampling

> **Key finding:** Representativeness, not domain knowledge, determines sampler quality.
> Schema-free random uniform or cluster sampling dominates all proxy-guided methods
> (DAG-MCMC, importance-weighted, stratified) on every real dataset tested.

---

## Abstract

Systematic data quality profiling — computing missing-value rates, duplicate fractions,
outlier densities, and functional-dependency violations — is foundational for data-centric
AI pipelines, yet exhaustive scans over millions of rows are prohibitively slow for
near-real-time monitoring.

We benchmark nine progressive sampling strategies on three real administrative datasets
(NYC 311, NYPD arrests, UCI Adult; up to 500K rows), an IoT sensor stream (2.3M rows), and
two ultra-large datasets (up to 7.4M rows), plus synthetic data scaled to 5×10⁶ rows.
At a 5% budget, random uniform achieves 0.49% mean relative error on NYC 311; DAG-guided
MCMC yields 19.5% (40× worse). Cluster sampling matches random uniform with no added
complexity. At scale, random uniform is near-linear (O(N^0.964)) while DAG is super-linear
(O(N^1.272)), running 28–47× slower with 6× worse accuracy on ultra-large data.



---

## Sampling Strategies Benchmarked

| Strategy | Type | Key parameter |
|----------|------|--------------|
| `random_uniform` | Blind | None |
| `geometric` | Blind | Growth factor r |
| `yamane` | Blind | Yamane formula |
| `cluster` | Blind | k = max(10, √N) blocks |
| `metropolis_hastings` | Proxy-guided | IQR proxy |
| `dag` | Proxy-guided | Dependency graph |
| `strat_column` | Proxy-guided | 3-stratum column type |
| `strat_quality` | Proxy-guided | 3-stratum quality score |
| `importance` | Proxy-guided | Weighted reservoir ∝ proxy |

---

## Datasets

| ID | Source | Rows | Cols | Role |
|----|--------|------|------|------|
| D1 | Synthetic (numpy) | 100K | 8 | Controlled — E1, E2, E4 |
| D2 | [NYC 311 Service Requests](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9) | 500K | 9 | Real admin — E1, E3 |
| D3 | [NYPD Arrests](https://data.cityofnewyork.us/Public-Safety/NYPD-Arrests-Data-Historic-/8h9b-rp9u) | 500K | 18 | Real admin — E1, E3 |
| D4 | [UCI Adult Census](https://archive.ics.uci.edu/dataset/2/adult) | 48K | 14 | Real census — E1, E3 |
| D5 | Synthetic (SDV or numpy fallback) | 10K–5M | 15 | Scalability — E5 |
| D6-A | [Ultra-Marathon Running](https://www.kaggle.com/datasets/aiaiaidavid/the-big-dataset-of-ultra-marathon-running) | 7.4M | 11 | XXL real — E6 |
| D6-B | [NYC Yellow Taxi Jan–Feb 2023](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) | 6.0M | 19 | XXL real — E6 |
| D7 | [Intel Lab IoT](http://db.csail.mit.edu/labdata/) | 2.3M | 5 | IoT sensor — E8 |

**Preprocessed parquet files for D2, D3, D4, D7 are included in `data/`.**
D6-A and D6-B must be downloaded separately (see instructions below).

---

## Setup

**Requirements:** Python ≥ 3.10 (tested on 3.13).

```bash
git clone https://github.com/<your-repo>/profiling-benchmark.git
cd profiling-benchmark
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **SDV note (scalability experiment):** SDV requires pandas < 3 and a compatible
> torch wheel, which is unavailable on Python 3.13 / macOS x86_64. Correlation experiment automatically
> falls back to correlated numpy data — results are equivalent. To use real SDV:
> run on Python ≤ 3.12 or Apple Silicon, then `pip install "pandas>=2.2.3,<3" sdv`.

### Downloading XXL datasets 

```bash
# D8-A: Ultra-Marathon (~7.4M rows)
# Download TWO_CENTURIES_OF_UM_RACES.csv from Kaggle, then:
mv TWO_CENTURIES_OF_UM_RACES.csv data/ultra-marathon.zip   # or extract directly

# D8-B: NYC Yellow Taxi 2023
wget https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet -P data/
wget https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-02.parquet -P data/
```

Ground-truth profiles for D6-A and D6-B are computed on first run and cached
automatically to `data/D6_*.gt.json` (one-time cost of a few minutes).

---

## Reproducing Experiments

> **zsh shell note:** always quote array arguments to prevent glob expansion:
> ```bash
> python run_exp_E1.py 'experiment.seeds=[42,123,456]'  # correct
> python run_exp_E1.py experiment.seeds=[42,123,456]    # wrong — zsh expands [...]
> ```

All scripts use [Hydra](https://hydra.cc) for configuration.
Default configs are in `conf/`. Override any parameter via dot-notation on the command line.

### Full reproduction (all experiments)

```bash
# Smoke test first (fast, ~2 min total)
python run_exp_E1.py data.n_records=10000 'experiment.seeds=[42,123,456]'
python run_exp_E3.py data.n_records=10000 'experiment.seeds=[42,123,456]'

# Full runs (use default configs)
python run_exp_E1.py          # E1 — accuracy comparison          (C3)
python run_exp_E2.py          # E2 — correlation ablation         (C6)
python run_exp_E3.py          # E3 — budget–accuracy trade-off    (C2)
python run_exp_E4.py          # E4 — robustness under injection    (C5)
python run_exp_E5.py          # E5 — scalability                  (C4)
python run_exp_new_samplers.py  # E1+E3 for cluster/strat/importance samplers
```

### E6 — XXL real-dataset scalability 

Run one dataset at a time (each takes 30–60 min on a laptop):

```bash
python run_exp_E6.py experiment.dataset=ultra_marathon
python run_exp_E6.py experiment.dataset=nyc_taxi
```

Results → `outputs/E6/E6_results_ultra_marathon.csv` and `outputs/E6/E6_results_nyc_taxi.csv`.

### E7a / E7b — Ablations

```bash
python run_exp_E7a.py         # E7a — budget sensitivity ablation
python run_exp_E7b.py         # E7b — seed sensitivity ablation
```

### E8 — IoT sensor benchmark 

Requires D7 (Intel Lab data, included in `data/`):

```bash
python run_exp_E8.py          # if present, else included in run_exp_new_samplers.py
```


## Project Structure

```
profiling/                Core library
  samplers.py             Nine sampling strategy implementations
  dag.py                  DAG-MCMC sampler
  convergence.py          Convergence criteria
  progressive.py          Progressive profiling loop
  quality_profile.py      Quality indicator computation
  datasets.py             Dataset loaders
  data_generator.py       Synthetic data generation

conf/
  config.yaml             Root config (E1–E5, E7–E8)
  config_e6.yaml          Root config for E6 (XXL)
  experiment/             Per-experiment Hydra configs (E1.yaml … E8.yaml)

run_exp_E*.py             Experiment entry points
run_exp_new_samplers.py   Runner for cluster/stratified/importance samplers

data/                     Preprocessed datasets (not included in this repo)

```

---

## Citation

If you use this benchmark or code, please cite:

```bibtex
@article{bertie2026profiling,
  author    = {Laure Berti-Equille},
  title     = {Data Quality Profiling at Scale with Progressive Sampling:
               A Benchmark for Data-Centric {AI} Pipelines},
  note      = {Under submission},
  year      = {2026}
}
```

---

## License

Code: MIT License.
Data: see individual dataset sources above for their respective licenses.

---

## Contact

Laure Berti-Equille — [laure.berti@ird.fr](mailto:laure.berti@ird.fr) — IRD, ESPACE-DEV, Montpellier, France
[Web page](https://laureberti.github.io/website/)
