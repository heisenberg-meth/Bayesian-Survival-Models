It's public now. Let me pull it apart.This is a much more mature repo than the checklist suggested — let's see what's actually done.Good — it's public now and I pulled the actual code, commit history, reports, and figures. This changes the picture significantly: **this project is far more built-out than your checklist suggested.** Let me give you the real status.

## Actual status: substantially complete, not "in development" in the way the checklist implied

Looking at the commit history (3 real feature commits between Jul 27 and today, Aug 6), the repo already has:

- **All 4 model families implemented and run on all 3 datasets** (GBSG2, WHAS500, METABRIC) — KM plots (stratified by multiple covariates, not just overall), log-rank benchmarking (`scripts/benchmark_log_rank.py`), Cox PH with forest plots, RSF with feature importance + calibration, DeepSurv with training curves + calibration, Bayesian Cox with posterior HR plots and credible survival bands
- **A full evaluation/statistical layer you hadn't listed**: `evaluate.py`, `run_statistical_validation.py` (Friedman test, pairwise Wilcoxon with Bonferroni/Holm/FDR-BH correction, bootstrap CIs, stability across seeds, sensitivity to training size, runtime profiling)
- **Explainability**: SHAP analysis, permutation importance, posterior analysis (`src/explainability/`)
- **Extensions beyond your checklist**: censoring robustness, small-dataset analysis, ablation study (`run_extensions.py`)
- **A draft manuscript**: `reports/paper/comparative_survival_analysis_paper.md`

So in terms of your original 6-item list, **every item has real, executed output with figures and reports**, not just a script that runs. That's good news — you're much closer to done than "add upcoming things" suggested.

## But I found three things worth fixing before calling it finished

I actually read the code, not just the file tree, and found real problems:

**1. The Bayesian Cox numbers look wrong, not just "trade-off."**
`model_comparison_report.md` shows Bayesian Cox with IBS around 0.38–0.44 while every other model sits at 0.14–0.24 — a 2–3x gap. The report explains this away as a piecewise-hazard vs. continuous-hazard trade-off, but the posterior estimates back up a different explanation: in `bayesian_cox_report.md`, nearly every feature's hazard ratio has **Prob(HR > 1) ≈ 0.45–0.55** with huge credible intervals (e.g. 0.28 to 3.3). That's what a posterior looks like when it hasn't moved away from the prior — i.e., **the model isn't learning from data**, not "calibration trade-off." Two candidate causes I'd check in `src/models/bayesian/model.py`:

- `n_advi_iterations=1200` is low for ADVI to converge on a model with piecewise hazards + regression coefficients together
- `chains=1` is the default — with a single MCMC chain you can't compute a meaningful R̂ (convergence diagnostic needs ≥2 chains), so `diagnostics.py`'s R̂ output may be uninformative even when NUTS is used

I'd fix this before writing the "Bayesian Cox adds calibrated uncertainty" conclusion into your paper draft — right now the evidence doesn't support the "regularization benefit" claim in section 2 of `model_comparison_report.md` as strongly as it's written.

**2. The pipeline/CLI abstraction layer is dead scaffolding.**
`docs/architecture.md` describes `src/pipelines/` as orchestrating training, evaluation, and inference — but all four classes in `src/pipelines/` are stubs: `EvaluationPipeline.run()` literally returns a hardcoded `{"c_index": 0.75, "integrated_brier_score": 0.12}`, `TrainingPipeline.run()` returns `None`, and `main.py` just prints "Hello from bayesian-cox!" All your real work happens through the standalone `scripts/*.py` files instead, which is fine functionally, but the architecture doc oversells what exists. Either wire up the pipeline layer for real, or delete it and update the doc — right now it's misleading if anyone else reads the architecture doc expecting a working API.

**3. Test suite is mostly empty.**
`tests/test_data.py`, `test_metrics.py`, `test_pipeline.py`, and `test_training.py` are all **0 lines**. Only `test_models.py` (155 lines) has real content. `run_tests.py` will happily report "0 run, 0 failed" for those files and look like a pass. This is the gap most likely to bite you if you touch preprocessing or metrics code later without noticing a regression.

## Updated completion checklist

- [ ] Investigate and fix the Bayesian Cox convergence issue (increase ADVI iterations, run ≥2 chains for real R̂, or switch default to NUTS and check ESS) before trusting the reported numbers
- [ ] Rewrite the "calibration trade-off" framing in `model_comparison_report.md` / `bayesian_cox_report.md` once the real cause is known
- [ ] Either implement `src/pipelines/*` for real or remove them + update `architecture.md` to match reality
- [ ] Fill in `test_data.py`, `test_metrics.py`, `test_pipeline.py`, `test_training.py` (or delete them if genuinely out of scope)
- [ ] Update `docs/experiment_log.md` — it still shows all 4 entries as "Initialized" with blank C-Index/IBS, dated Jul 27, even though results have existed since then
- [ ] Decide on Survival SVM — given RSF + DeepSurv already cover ML/DL and you have statistical validation infra built, I'd skip it; it adds a 5th model without a clear question it answers that RSF doesn't
- [ ] Finish the paper draft's discussion section once #1 is resolved, since the current conclusion may need revising

## Commands (matches what's actually in the repo)

```bash
# environment (README recommends conda due to PyMC/torchtuples pinning)
conda env create -f environment.yml
conda activate survival-benchmark
# — or pip —
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# data
python scripts/download_datasets.py
python scripts/prepare_dataset.py

# EDA + log-rank (your KM/log-rank items)
python scripts/run_eda.py
python scripts/benchmark_log_rank.py

# the four models
python scripts/run_cox_ph.py
python scripts/run_rsf.py
python scripts/run_deepsurv.py
python scripts/run_bayesian_cox.py


# evaluation, stats, explainability, extensions — all real, all already in the repo
python scripts/evaluate.py
python scripts/run_statistical_validation.py
python scripts/explain.py
python scripts/run_extensions.py

# tests
python tests/run_tests.py
```

If you want, I can go into `src/models/bayesian/model.py` and `sampler.py` in more depth and pin down exactly why the posteriors aren't converging — that's the one thing here that actually affects your conclusions, not just project hygiene.

Here's the finish list, ordered by what actually blocks calling this done — not by file location.

## Must-fix (these affect your results/conclusions, not just cleanliness)

1. **Fix the Bayesian Cox convergence problem.** This is the real blocker. Right now `Prob(HR>1) ≈ 0.5` for nearly every top feature and IBS is 2–3x worse than every other model — that's the signature of a posterior that never moved off the prior, not a genuine "calibration trade-off" as the report claims. Concretely:
   - Bump `n_advi_iterations` well past 1200 and check the ELBO convergence curve actually flattens
   - Run `chains=2` or more for NUTS/MCMC — with `chains=1` your R̂ diagnostic in `diagnostics.py` is meaningless
   - Check ESS (effective sample size), not just that it "ran"
2. **Rewrite the Bayesian Cox conclusions once the above is fixed.** `model_comparison_report.md`, `bayesian_cox_report.md`, and the paper draft's discussion section all currently assert a "regularization benefit" that the numbers don't support yet. Don't publish/finalize the paper draft until this is re-run and re-written.

## Should-fix (real gaps, lower stakes)

3. **Decide the fate of `src/pipelines/`.** They're stubs (`EvaluationPipeline.run()` returns a hardcoded `{"c_index": 0.75, ...}`) but `architecture.md` describes them as the real orchestration layer. Either implement them for real, or delete them and correct the doc — don't leave the mismatch.
4. **Write the missing tests.** `test_data.py`, `test_metrics.py`, `test_pipeline.py`, `test_training.py` are all empty (0 lines). `run_tests.py` will silently report them as passing with 0 tests run. At minimum, cover the preprocessing pipeline and metrics computation — that's where a silent bug would hurt most.
5. **Update `docs/experiment_log.md`.** Still shows all 4 rows as "Initialized" with blank C-Index/IBS from Jul 27, despite results existing since then. Low effort, makes the repo look unfinished when it isn't.

## Decide and move on (don't let these drag)

6. **Survival SVM** — skip it. RSF + DeepSurv already cover the ML/DL angle, and adding a 5th model doesn't answer a new question. Note it as "considered, out of scope" in the README rather than leaving it open.
7. **Finalize the paper draft** — but only after #1 and #2 are done, since the discussion section depends on correct Bayesian numbers.

If you want, tell me which one you want to tackle first and I'll go deep on just that — the Bayesian convergence fix is the one with the most actual technical work in it.

Dug into the actual code. Found the root cause — and it's more concrete than I could tell from the reports alone.

## Root cause of the Bayesian Cox problem

`config/model.yaml` specifies real, sensible hyperparameters for the Bayesian Cox model:

```yaml
bayesian_cox:
  beta_prior_mean: 0.0
  beta_prior_sd: 10.0
  mcmc:
    draws: 2000
    tune: 1000
    chains: 4
    target_accept: 0.95
```

But `scripts/run_bayesian_cox.py` **never loads this file**. It defines its own separate CLI defaults that quietly override everything in the YAML:

```python
parser.add_argument("--draws", type=int, default=400, ...)
parser.add_argument("--tune", type=int, default=300, ...)
parser.add_argument("--chains", type=int, default=1, ...)
```

And the prior width defaults to `sigma=1.0` inside `BayesianCoxModel` (via `prior_params.get("sigma", 1.0)`), not the `beta_prior_sd: 10.0` from config. So what actually ran was: **1 chain, 400 draws, 300 tune, prior sigma=1** — instead of the intended 4 chains / 2000 draws / 1000 tune / sigma=10.

That combination explains everything you're seeing:

- **1 chain** → no real R̂ can be computed (Gelman-Rubin needs ≥2 chains), so you can't actually tell if it converged
- **A prior 10x tighter than intended** (σ=1 vs σ=10) pulls every coefficient toward 0
- **Short sampling** on top of a tight prior → the posterior barely moves from the prior → `Prob(HR>1) ≈ 0.5` and wide credible intervals for nearly every feature — that's the visible symptom
- This directly explains the inflated IBS too: a model whose coefficients are shrunk near zero produces flat, uninformative survival curves, which score badly on Brier score regardless of the piecewise-hazard framing in the report

There's also a `get_project_config()` utility in `src/utils/config.py` that's fully built (loads and merges `base.yaml`, `datasets.yaml`, `model.yaml`, `training.yaml`) — and I checked: **`run_cox_ph.py` and `run_rsf.py` don't call it either.** So this isn't a Bayesian-specific bug, it's a repo-wide pattern: the config system exists but nothing wires into it. Every script's actual behavior comes from its own CLI argparse defaults, and the YAML files are essentially decorative right now.

## What to actually do

1. **For the Bayesian model specifically**: re-run with `--chains 4 --draws 2000 --tune 1000` and pass `--prior_params '{"sigma": 10.0}'` (or just wire the config properly — see #2), then check R̂ and ESS via `get_mcmc_diagnostics()`, which already exists and works, it's just never been fed a multi-chain result to diagnose.
2. **Fix config wiring properly** (better than patching CLI defaults one-off): have each script call `get_project_config()` and use it as the default, with CLI args only as overrides. This fixes cox_ph and rsf too, not just Bayesian — right now none of them are provably using the hyperparameters documented in `config/model.yaml`, so anyone reading the config to understand what ran is being misled.
3. **Re-run the full comparison** once fixed, and only then update `model_comparison_report.md` / `bayesian_cox_report.md` / the paper draft's discussion section.

Want me to write the config-loading fix for `run_bayesian_cox.py` (and the other two scripts) so you can just re-run it?

# PRD — Config Wiring & Bayesian Cox Convergence Fix

**Project:** Bayesian-Survival-Models
**Status:** Draft
**Owner:** heisenberg-meth
**Prepared:** 2026-08-06

---

## 1. Problem Statement

The repository has a working, fully-featured configuration system (`src/utils/config.py` → `get_project_config()`) and four YAML files (`config/base.yaml`, `datasets.yaml`, `model.yaml`, `training.yaml`) that document intended hyperparameters. **None of the four model scripts actually load or use this system.** Each script hardcodes its own defaults directly in code, which silently diverge from the documented config. This has two consequences:

1. **Silent misconfiguration**: anyone reading `config/model.yaml` to understand what ran is reading fiction. The values there are not what produced the results in `reports/`.
2. **A real correctness bug in Bayesian Cox**: the divergence between intended and actual hyperparameters (1 chain instead of 4, sigma=1.0 instead of 10.0, ~400 draws instead of 2000) is the most likely root cause of the anomalous results in `reports/bayesian_cox_report.md` and `reports/model_comparison_report.md` — specifically `Prob(HR>1) ≈ 0.5` for nearly every feature and IBS 2–3x worse than every other model.

This PRD scopes the fix for both problems: wire config properly, and re-validate Bayesian Cox once it's running with the intended settings.

---

## 2. Goals

- Every model script reads its hyperparameters from `config/model.yaml` (and `config/training.yaml` where relevant), with CLI arguments acting only as optional overrides — never as the source of truth.
- Bayesian Cox runs with enough chains, draws, and the correct prior width to produce a trustworthy posterior, with real R̂/ESS diagnostics to prove it.
- All reports and the paper draft that depend on Bayesian Cox numbers are regenerated and rewritten once the fix lands — no report should describe uninvestigated behavior as an intentional "trade-off."

## 3. Non-Goals

- No new model families (Survival SVM explicitly out of scope — already decided).
- No change to dataset preprocessing, splits, or evaluation metrics themselves.
- No rewrite of `src/pipelines/*` — tracked as a separate, lower-priority item (see §7).

---

## 4. Affected Scripts & Files

| File                                                                                                                                 | Current behavior                                                                                                                                                                                                                              | Required change                                                                                                                                      |
| ------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `scripts/run_bayesian_cox.py`                                                                                                      | Own argparse defaults:`--draws 400 --tune 300 --chains 1`; `prior_params` defaults to `sigma=1.0` inside `BayesianCoxModel`, ignoring `beta_prior_sd: 10.0` in config                                                               | Load`config/model.yaml → bayesian_cox` as defaults; CLI args override only if explicitly passed                                                   |
| `scripts/run_cox_ph.py`                                                                                                            | Hardcodes`CoxPHModel(l2_reg=1e-4)`, no relation to `config/model.yaml`'s `penalizer: 0.1` / `l1_ratio: 0.0` (which don't even map to `CoxPHModel`'s actual constructor param `l2_reg`)                                            | Load config; also fix`config/model.yaml`'s `cox_ph` section to match the real constructor signature (`l2_reg`, not `penalizer`/`l1_ratio`) |
| `scripts/run_rsf.py`                                                                                                               | Own internal`param_grid` (n_estimators 40–75, max_depth 4–6, min_samples_leaf 3–5, min_samples_split 6–10), unrelated to config's single-point values (`n_estimators: 100, max_depth: 8, min_samples_split: 10, min_samples_leaf: 5`) | Load config values as the grid's center point or default; keep grid search but seed it from config, not arbitrary hardcoded ranges                   |
| `scripts/run_deepsurv.py`                                                                                                          | Own internal grid:`hidden_dims` in `{[16,8], [32,16]}`, `l2_reg` in `{1e-4,1e-3,1e-2}` — no relation to config's `hidden_dims: [64,32], dropout: 0.2, learning_rate: 0.001`                                                        | Load config as default single-run config; keep the grid search as an optional`--tune` flag, not the only path                                      |
| `src/models/bayesian/model.py`                                                                                                     | Accepts`chains`, `draws`, `tune`, `prior_params` as constructor args — this part is fine, it's a config-consumer, not the bug                                                                                                        | No change required, just needs to receive the right values                                                                                           |
| `src/utils/config.py`                                                                                                              | `get_project_config()` fully implemented and correct                                                                                                                                                                                        | No change required, just needs to be called                                                                                                          |
| `config/model.yaml`                                                                                                                | Values for`cox_ph` don't match the actual model's constructor signature                                                                                                                                                                     | Fix`cox_ph` section keys to match `CoxPHModel.__init__`                                                                                          |
| `docs/experiment_log.md`                                                                                                           | All 4 rows stuck at "Initialized" with blank metrics since 2026-07-27, despite results existing                                                                                                                                               | Update rows with real status + metrics once each script is re-run                                                                                    |
| `reports/bayesian_cox_report.md`, `reports/model_comparison_report.md`, `reports/paper/comparative_survival_analysis_paper.md` | Describe current Bayesian Cox numbers as a deliberate calibration trade-off                                                                                                                                                                   | Rewrite once real posteriors are obtained under corrected config                                                                                     |

---

## 5. Detailed Requirements

### 5.1 Config loader integration (all four `run_*.py` scripts)

- At the top of each script, call `cfg = get_project_config()` and read the relevant model sub-block, e.g. `cfg.models.bayesian_cox`.
- Every CLI `argparse` flag's `default=` must pull from `cfg` instead of a literal, e.g.:
  ```python
  parser.add_argument("--chains", type=int, default=cfg.models.bayesian_cox.mcmc.chains)
  ```
- If `config/model.yaml` is missing or a key is absent, fall back to the current hardcoded literal (don't hard-fail existing runs).
- Log the resolved configuration (source: config file vs CLI override) at the start of each run so reproducibility is auditable from stdout/logs alone.

**Acceptance criteria:**

- Running any of the four scripts with no CLI flags produces the exact hyperparameters listed in `config/model.yaml`.
- Passing an explicit CLI flag overrides the config value, and this is visible in the printed run header.

### 5.2 Fix `config/model.yaml` → `cox_ph` section

- Current keys (`penalizer`, `l1_ratio`) don't exist on `CoxPHModel.__init__(self, l2_reg=1e-4)`.
- Replace with the correct key: `l2_reg: 0.0001` (or a deliberately chosen value — see §5.4).

**Acceptance criteria:** `config/model.yaml`'s `cox_ph` section has keys that map 1:1 to `CoxPHModel.__init__`'s actual parameters.

### 5.3 Bayesian Cox re-run under corrected settings

- Re-run `scripts/run_bayesian_cox.py` for all three datasets (GBSG2, WHAS500, METABRIC) with:
  - `chains=4`, `draws=2000`, `tune=1000` (per `config/model.yaml`)
  - `beta_prior_sd=10.0` (per config) — or, if this was intentionally tightened for a reason not documented anywhere, that reasoning must be written into `config/model.yaml` as a comment and into `bayesian_cox_report.md`
  - `target_accept=0.95` if using NUTS (already in config, currently unused by the script)
- Call `get_mcmc_diagnostics()` (already implemented in `src/models/bayesian/model.py`) after fitting and persist R̂ and ESS per feature to `reports/tables/bayesian_cox_results.json`.

**Acceptance criteria:**

- R̂ < 1.01 for all `beta` parameters on all three datasets (standard convergence threshold).
- ESS (bulk) > 400 for all `beta` parameters.
- If either threshold fails after the above settings, escalate: increase `tune`/`draws` further, or switch to NUTS with `target_accept=0.95` if currently on ADVI, before concluding the model itself is misspecified.

### 5.4 Re-evaluate and rewrite dependent reports

Only after §5.3 passes its acceptance criteria:

- Regenerate `reports/tables/bayesian_cox_results.json`, `reports/tables/model_comparison.json`, `reports/tables/model_rankings.csv` via `scripts/evaluate.py` and `scripts/run_statistical_validation.py`.
- Rewrite `reports/bayesian_cox_report.md` §2 (posterior tables) and §3 (four-model comparison) with the new numbers.
- Rewrite `reports/model_comparison_report.md` §4 ("Key Synthesis & Insights") — specifically item 4, which currently attributes elevated IBS to a "calibration trade-off." Replace with whatever the corrected run actually shows. If IBS is still elevated after convergence is confirmed, that conclusion becomes defensible; if it drops to be competitive, say so instead.
- Update `reports/paper/comparative_survival_analysis_paper.md`'s discussion section to match.

**Acceptance criteria:** No report or paper section describes Bayesian Cox behavior that hasn't been verified against a converged model (R̂/ESS checked, not assumed).

### 5.5 Housekeeping

- Update `docs/experiment_log.md`: replace the four "Initialized" rows with real dates, status (`Complete`), and actual C-Index/IBS values once each script has been re-run under the fixed config.
- Add a one-line note to `README.md`'s reproducibility section stating that hyperparameters are sourced from `config/*.yaml`, not script defaults, so future readers don't hit the same confusion.

---

## 6. Rollout Order

1. Fix `config/model.yaml`'s `cox_ph` section (§5.2) — trivial, no dependencies.
2. Wire config loading into all four scripts (§5.1).
3. Re-run `run_bayesian_cox.py` under corrected settings, verify convergence (§5.3). **Do not proceed past this step until R̂/ESS pass.**
4. Re-run `run_cox_ph.py`, `run_rsf.py`, `run_deepsurv.py` under config-driven defaults to confirm no unintended regression from wiring config (these should reproduce close to existing numbers, since their config values are reasonable — this step is a sanity check, not expected to change conclusions).
5. Regenerate evaluation/statistical validation artifacts (§5.4).
6. Rewrite affected reports and paper draft (§5.4).
7. Update `docs/experiment_log.md` and `README.md` (§5.5).

## 7. Explicitly Deferred (tracked, not in this PRD's scope)

- `src/pipelines/*` stub implementations vs. `docs/architecture.md`'s description of them as the real orchestration layer.
- Empty test files: `tests/test_data.py`, `tests/test_metrics.py`, `tests/test_pipeline.py`, `tests/test_training.py`.
- Survival SVM — decided out of scope, not revisited unless a specific new question arises that RSF/DeepSurv don't answer.

## 8. Success Criteria (project-level)

This PRD is done when:

- [ ] `config/model.yaml` values are provably what's running (verified via printed run headers, not assumed)
- [ ] Bayesian Cox posteriors converge (R̂ < 1.01, ESS > 400) on all three datasets
- [ ] All reports referencing Bayesian Cox performance reflect the converged model's actual output
- [ ] `docs/experiment_log.md` no longer shows stale "Initialized" placeholder rows
