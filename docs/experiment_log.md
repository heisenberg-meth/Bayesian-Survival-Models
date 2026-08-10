# Experiment Log

> **Note:** All experiments were initially run with script-local defaults that diverged from `config/model.yaml`.
> As of 2026-08-06, config wiring has been fixed — scripts now read hyperparameters from `config/*.yaml`.
> Results below reflect the *original* runs. A full re-run under corrected config is pending.

| Experiment ID | Date | Dataset | Model | C-Index | IBS | Status | Notes |
|---|---|---|---|---|---|---|---|
| EXP-001 | 2026-07-27 | GBSG2 | Cox PH Baseline | — | — | Complete (pre-config-fix) | L2=1e-4 (script default, matched config). Pending re-run for consistency verification. |
| EXP-002 | 2026-07-27 | GBSG2 | Bayesian Cox | — | — | Complete (pre-config-fix) | **Config mismatch identified:** ran with draws=400, tune=300, chains=1, sigma=1.0 instead of config's draws=2000, tune=1000, chains=4, sigma=10.0. Results unreliable — pending re-run. |
| EXP-003 | 2026-07-27 | WHAS500 | Bayesian Cox | — | — | Complete (pre-config-fix) | Same config mismatch as EXP-002. Pending re-run. |
| EXP-004 | 2026-07-27 | METABRIC | Bayesian Cox / DeepSurv | — | — | Complete (pre-config-fix) | Same config mismatch for Bayesian Cox. DeepSurv used script-local grid, not config. Pending re-run. |
| EXP-005 | 2026-07-27 | All 3 | RSF | — | — | Complete (pre-config-fix) | Grid search used arbitrary ranges, not config-seeded. Pending re-run. |
| EXP-006 | 2026-08-06 | — | — | — | — | Config Fix Applied | All 4 scripts now load `config/model.yaml` via `get_project_config()`. CLI args override only when explicitly passed. |
