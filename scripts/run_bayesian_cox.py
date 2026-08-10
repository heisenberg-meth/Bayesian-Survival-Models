"""
Master Execution Script for Phase 9 — Bayesian Cox Model (PyMC MCMC / ADVI).
Trains Bayesian Cox models across GBSG2, WHAS500, and METABRIC preprocessed datasets.
Evaluates posterior hazard ratio distributions, 95% credible intervals, test discrimination (C-index)
and calibration (IBS), executes 5-fold CV, compares against Cox PH, RSF, and DeepSurv baselines,
exports results, and generates publication plots.

Hyperparameters are loaded from config/model.yaml; CLI arguments override when explicitly passed.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.cross_validation import CrossValidationEvaluator
from src.evaluation.metrics import evaluate_survival_model
from src.models.bayesian.model import BayesianCoxModel
from src.utils.config import get_project_config

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
TABLES_DIR = os.path.join(REPORTS_DIR, "tables")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Load project config — canonical hyperparameter source
# ---------------------------------------------------------------------------
def _load_model_config():
    """Load Bayesian Cox config from config/model.yaml, with safe fallbacks."""
    try:
        cfg = get_project_config(os.path.join(PROJECT_ROOT, "config"))
        bcfg = cfg.models.bayesian_cox
    except Exception:  # noqa: BLE001
        # Fallback if config files are missing
        from src.utils.config import ConfigDict

        bcfg = ConfigDict(
            {
                "beta_prior_mean": 0.0,
                "beta_prior_sd": 10.0,
                "coefficient_prior": "normal",
                "n_intervals": 6,
                "mcmc": {
                    "draws": 2000,
                    "tune": 1000,
                    "chains": 4,
                    "target_accept": 0.95,
                },
                "advi": {"n_iterations": 10000},
            }
        )
    return bcfg


def draw_posterior_hr_forest_plot(summary_df, dataset_name, filepath):
    """Draws Bayesian Posterior Hazard Ratio Forest Plot with 95% Credible Intervals using Pillow."""
    img_w, img_h = 900, 600
    img = Image.new("RGB", (img_w, img_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    pad_left, pad_right, pad_top, pad_bot = 220, 50, 70, 60
    plot_w = img_w - pad_left - pad_right
    plot_h = img_h - pad_top - pad_bot

    draw.text(
        (img_w // 2 - 220, 20),
        f"Bayesian Posterior Hazard Ratios — {dataset_name.upper()}",
        fill=(30, 41, 59),
    )
    draw.text(
        (img_w // 2 - 80, img_h - 35),
        "Hazard Ratio (Posterior 95% Credible Interval)",
        fill=(51, 65, 85),
    )

    n_feats = len(summary_df)
    row_h = plot_h / max(1, n_feats)

    all_lows = summary_df["95% Credible Lower"].values
    all_ups = summary_df["95% Credible Upper"].values
    min_hr = min(0.5, float(np.min(all_lows)) * 0.9)
    max_hr = max(2.5, float(np.max(all_ups)) * 1.1)

    def hr_to_x(hr_val):
        norm = (hr_val - min_hr) / (max_hr - min_hr)
        return pad_left + int(norm * plot_w)

    x_null = hr_to_x(1.0)
    draw.line(
        [(x_null, pad_top), (x_null, img_h - pad_bot)], fill=(220, 38, 38), width=2
    )
    draw.text((x_null - 15, pad_top - 20), "HR = 1.0", fill=(220, 38, 38))

    for hr_val in [0.5, 1.0, 1.5, 2.0, 2.5]:
        if min_hr <= hr_val <= max_hr:
            x_pos = hr_to_x(hr_val)
            draw.line(
                [(x_pos, pad_top), (x_pos, img_h - pad_bot)],
                fill=(241, 245, 249),
                width=1,
            )
            draw.text(
                (x_pos - 10, img_h - pad_bot + 10),
                f"{hr_val:.1f}",
                fill=(100, 116, 139),
            )

    for idx, row in summary_df.iterrows():
        feat = str(row["feature"])
        mean_hr = float(row["exp(coef) HR Mean"])
        low_hr = float(row["95% Credible Lower"])
        up_hr = float(row["95% Credible Upper"])

        y_pos = int(pad_top + idx * row_h + row_h / 2.0)
        draw.text((20, y_pos - 8), feat[:24], fill=(30, 41, 59))

        x_mean = hr_to_x(mean_hr)
        x_low = hr_to_x(low_hr)
        x_up = hr_to_x(up_hr)

        draw.line([(x_low, y_pos), (x_up, y_pos)], fill=(79, 70, 229), width=3)
        draw.line([(x_low, y_pos - 4), (x_low, y_pos + 4)], fill=(79, 70, 229), width=2)
        draw.line([(x_up, y_pos - 4), (x_up, y_pos + 4)], fill=(79, 70, 229), width=2)

        draw.ellipse(
            [(x_mean - 5, y_pos - 5), (x_mean + 5, y_pos + 5)], fill=(147, 51, 234)
        )
        draw.text(
            (x_up + 10, y_pos - 8),
            f"{mean_hr:.2f} [{low_hr:.2f}, {up_hr:.2f}]",
            fill=(51, 65, 85),
        )

    img.save(filepath)


def draw_bayesian_credible_survival_plot(
    eval_times, surv_mean, surv_low, surv_up, dataset_name, filepath
):
    """Draws Bayesian Survival Curves with 95% Credible Interval Shading using Pillow."""
    img_w, img_h = 900, 600
    img = Image.new("RGB", (img_w, img_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    pad_left, pad_right, pad_top, pad_bot = 80, 40, 70, 80
    plot_w = img_w - pad_left - pad_right
    plot_h = img_h - pad_top - pad_bot

    draw.text(
        (img_w // 2 - 240, 20),
        f"Bayesian Survival & 95% Credible Band — {dataset_name.upper()}",
        fill=(30, 41, 59),
    )
    draw.text((img_w // 2 - 50, img_h - 35), "Time Horizon", fill=(51, 65, 85))
    draw.text((15, img_h // 2 - 60), "Survival S(t)", fill=(51, 65, 85))

    draw.line(
        [(pad_left, pad_top), (pad_left, img_h - pad_bot)],
        fill=(148, 163, 184),
        width=2,
    )
    draw.line(
        [(pad_left, img_h - pad_bot), (img_w - pad_right, img_h - pad_bot)],
        fill=(148, 163, 184),
        width=2,
    )

    for y_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y_pos = img_h - pad_bot - int(y_val * plot_h)
        draw.line(
            [(pad_left, y_pos), (img_w - pad_right, y_pos)],
            fill=(226, 232, 240),
            width=1,
        )
        draw.text((pad_left - 45, y_pos - 8), f"{y_val:.2f}", fill=(71, 85, 105))

    max_t = max(eval_times) if len(eval_times) > 0 else 1.0

    pts_mean, pts_low, pts_up = [], [], []
    for t, m_s, l_s, u_s in zip(eval_times, surv_mean, surv_low, surv_up):
        x_pos = pad_left + int((t / max_t) * plot_w)
        y_m = img_h - pad_bot - int(m_s * plot_h)
        y_l = img_h - pad_bot - int(l_s * plot_h)
        y_u = img_h - pad_bot - int(u_s * plot_h)

        pts_mean.append((x_pos, y_m))
        pts_low.append((x_pos, y_l))
        pts_up.append((x_pos, y_u))

    for i in range(len(pts_mean) - 1):
        draw.line([pts_low[i], pts_low[i + 1]], fill=(199, 210, 254), width=2)
        draw.line([pts_up[i], pts_up[i + 1]], fill=(199, 210, 254), width=2)
        draw.line([pts_mean[i], pts_mean[i + 1]], fill=(79, 70, 229), width=3)

    leg_x = img_w - pad_right - 240
    leg_y = pad_top + 20
    draw.line(
        [(leg_x, leg_y + 5), (leg_x + 20, leg_y + 5)], fill=(79, 70, 229), width=3
    )
    draw.text((leg_x + 30, leg_y), "Posterior Mean S(t)", fill=(30, 41, 59))

    draw.line(
        [(leg_x, leg_y + 30), (leg_x + 20, leg_y + 30)], fill=(199, 210, 254), width=2
    )
    draw.text((leg_x + 30, leg_y + 25), "95% Credible Interval", fill=(71, 85, 105))

    img.save(filepath)


def load_json_if_exists(path: str) -> dict:
    if not os.path.exists(path):
        return {}

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    # -----------------------------------------------------------------------
    # Load canonical config from config/model.yaml
    # -----------------------------------------------------------------------
    bcfg = _load_model_config()
    mcmc_cfg = bcfg.get("mcmc", {})
    advi_cfg = bcfg.get("advi", {})

    # Config-sourced defaults (CLI overrides only when explicitly passed)
    cfg_draws = mcmc_cfg.get("draws", 2000)
    cfg_tune = mcmc_cfg.get("tune", 1000)
    cfg_chains = mcmc_cfg.get("chains", 4)
    cfg_target_accept = mcmc_cfg.get("target_accept", 0.95)
    cfg_advi_iters = advi_cfg.get("n_iterations", 10000)
    cfg_prior = bcfg.get("coefficient_prior", "normal")
    cfg_intervals = bcfg.get("n_intervals", 6)
    cfg_prior_sigma = bcfg.get("beta_prior_sd", 10.0)
    cfg_prior_mu = bcfg.get("beta_prior_mean", 0.0)

    parser = argparse.ArgumentParser(
        description="Run Bayesian Cox Survival Model Execution Pipeline."
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["advi", "mcmc"],
        default="advi",
        help="Inference method to use: variational 'advi' or sampler 'mcmc'.",
    )
    parser.add_argument(
        "--prior",
        type=str,
        choices=["normal", "student-t", "laplace"],
        default=cfg_prior,
        help="Regression coefficient prior distribution type.",
    )
    parser.add_argument(
        "--prior-params",
        type=str,
        default=None,
        help="JSON string of parameter configurations for the prior (e.g. '{\"sigma\": 2.0}').",
    )
    parser.add_argument(
        "--draws",
        type=int,
        default=cfg_draws,
        help="Number of posterior samples to draw.",
    )
    parser.add_argument(
        "--tune", type=int, default=cfg_tune, help="Number of MCMC tuning iterations."
    )
    parser.add_argument(
        "--advi-iterations",
        type=int,
        default=cfg_advi_iters,
        help="Number of ADVI iterations.",
    )
    parser.add_argument(
        "--chains", type=int, default=cfg_chains, help="Number of MCMC chains to run."
    )
    parser.add_argument(
        "--intervals",
        type=int,
        default=cfg_intervals,
        help="Number of piecewise intervals for baseline hazard.",
    )
    parser.add_argument(
        "--target-accept",
        type=float,
        default=cfg_target_accept,
        help="Target acceptance rate for NUTS sampler.",
    )
    args = parser.parse_args()

    # Build prior_params: start from config defaults, overlay CLI JSON if given
    prior_params = {"mu": cfg_prior_mu, "sigma": cfg_prior_sigma}
    if args.prior_params:
        try:
            cli_prior = json.loads(args.prior_params)
            prior_params.update(cli_prior)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"[!] Error parsing --prior-params JSON string: {e}")
            sys.exit(1)

    # -----------------------------------------------------------------------
    # Log resolved configuration for reproducibility
    # -----------------------------------------------------------------------
    print(f"STARTING PHASE 9 — BAYESIAN COX MODEL ({args.method.upper()})")
    print("  Resolved Configuration (source: config/model.yaml + CLI overrides):")
    print(f"    Inference Method:    {args.method}")
    print(f"    Coefficient Prior:   {args.prior} (params: {prior_params})")
    print(f"    Piecewise Intervals: {args.intervals}")
    print(f"    Draws:               {args.draws}")
    print(f"    Tune:                {args.tune}")
    print(f"    Chains:              {args.chains}")
    print(f"    ADVI Iterations:     {args.advi_iterations}")
    print(f"    Target Accept:       {args.target_accept}")
    print("=" * 60)

    cox_results_path = os.path.join(TABLES_DIR, "cox_ph_results.json")
    rsf_results_path = os.path.join(TABLES_DIR, "rsf_results.json")
    deepsurv_results_path = os.path.join(TABLES_DIR, "deepsurv_results.json")

    cox_results = load_json_if_exists(cox_results_path)
    rsf_results = load_json_if_exists(rsf_results_path)
    deepsurv_results = load_json_if_exists(deepsurv_results_path)

    all_results = {}
    datasets = ["gbsg2", "whas500", "metabric"]

    for dname in datasets:
        print(f"\n[+] Fitting Bayesian Cox Model on Dataset: '{dname.upper()}'")
        dataset_dir = os.path.join(PROCESSED_DIR, dname)

        train_df = pd.read_csv(os.path.join(dataset_dir, "train.csv"))
        test_df = pd.read_csv(os.path.join(dataset_dir, "test.csv"))

        X_train = train_df.drop(columns=["time", "event"])
        y_train_time = train_df["time"].values
        y_train_event = train_df["event"].values

        X_test = test_df.drop(columns=["time", "event"])
        y_test_time = test_df["time"].values
        y_test_event = test_df["event"].values

        # 1. Fit Bayesian Cox Model — using config-sourced hyperparameters
        bayes_model = BayesianCoxModel(
            n_intervals=args.intervals,
            inference_method=args.method,
            n_advi_iterations=args.advi_iterations,
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            random_state=42,
            coefficient_prior=args.prior,
            prior_params=prior_params,
        )
        bayes_model.fit(X_train, y_train_time, y_train_event)

        summary_df = bayes_model.get_summary()
        print("\n    Posterior Hazard Ratio Summary (Top 5 Features):")
        print(
            summary_df.head(5)[
                [
                    "feature",
                    "exp(coef) HR Mean",
                    "95% Credible Lower",
                    "95% Credible Upper",
                    "Prob(HR > 1)",
                ]
            ].to_string(index=False)
        )

        # Print MCMC diagnostics if MCMC was run
        diag_list = []
        if args.method == "mcmc":
            diag_df = bayes_model.get_mcmc_diagnostics()
            print("\n    MCMC Convergence Diagnostics:")
            print(
                diag_df[
                    ["feature", "mean", "sd", "hdi_3%", "hdi_97%", "ess_bulk", "r_hat"]
                ].to_string(index=False)
            )
            diag_list = diag_df.to_dict(orient="records")

            # Check convergence criteria (R̂ < 1.01, ESS > 400)
            rhat_ok = all(diag_df["r_hat"] < 1.01)
            ess_ok = all(diag_df["ess_bulk"] > 400)
            print(
                f"\n    Convergence Check — R̂ < 1.01: {'PASS' if rhat_ok else 'FAIL'}"
            )
            print(f"    Convergence Check — ESS > 400: {'PASS' if ess_ok else 'FAIL'}")
            if not rhat_ok or not ess_ok:
                print(
                    "    [!] WARNING: Convergence criteria not met. Consider increasing draws/tune."
                )

        # 2. Test Evaluation
        eval_times = np.percentile(y_test_time, [25, 50, 75])
        test_risk = bayes_model.predict_risk(X_test)

        def surv_fn(times, bayes_model=bayes_model, X_test=X_test):
            return bayes_model.predict_survival(X_test, times)

        test_eval = evaluate_survival_model(
            y_time=y_test_time,
            y_event=y_test_event,
            risk_scores=test_risk,
            surv_prob_fn=surv_fn,
            eval_times=eval_times,
        )

        print(
            f"\n    Test C-Index:              {test_eval['c_index']} ± {test_eval['c_index_se']}"
        )
        print(
            f"    Test Integrated Brier:     {test_eval.get('integrated_brier_score', 'N/A')}"
        )

        # 3. 5-Fold Stratified Cross-Validation
        def model_trainer(df_tr, df_v):
            X_tr = df_tr.drop(columns=["time", "event"])
            y_tr_t = df_tr["time"].values
            y_tr_e = df_tr["event"].values
            m = BayesianCoxModel(
                n_intervals=args.intervals,
                inference_method=args.method,
                n_advi_iterations=args.advi_iterations // 2
                if args.method == "advi"
                else 600,
                draws=args.draws // 2 if args.method == "advi" else 150,
                tune=args.tune // 2 if args.method == "mcmc" else 150,
                chains=1,  # Speed up CV folds
                random_state=42,
                coefficient_prior=args.prior,
                prior_params=prior_params,
            )
            m.fit(X_tr, y_tr_t, y_tr_e)
            return m

        cv_evaluator = CrossValidationEvaluator(PROCESSED_DIR, dname)
        cv_results = cv_evaluator.evaluate_model(model_trainer, eval_times=eval_times)

        print(
            f"    5-Fold CV Mean C-Index:    {cv_results['mean_c_index']} ± {cv_results['std_c_index']}"
        )
        print(
            f"    5-Fold CV Mean IBS:        {cv_results.get('mean_integrated_brier_score', 'N/A')} ± {cv_results.get('std_integrated_brier_score', 'N/A')}"
        )

        # 4. Multi-Model Comparative Benchmarking
        cox_c = cox_results.get(dname, {}).get("test_eval", {}).get("c_index", "N/A")
        rsf_c = rsf_results.get(dname, {}).get("test_eval", {}).get("c_index", "N/A")
        ds_c = (
            deepsurv_results.get(dname, {}).get("test_eval", {}).get("c_index", "N/A")
        )

        print("\n    Four-Model Comparative Test C-Index:")
        print(f"    1. Cox PH Baseline:        {cox_c}")
        print(f"    2. Random Survival Forest: {rsf_c}")
        print(f"    3. DeepSurv Neural Net:    {ds_c}")
        print(f"    4. Bayesian Cox (PyMC):    {test_eval['c_index']}")

        # 5. Generate Visual Artifacts
        hr_plot_path = os.path.join(
            FIGURES_DIR, f"bayesian_cox_{dname}_posterior_hr.png"
        )
        draw_posterior_hr_forest_plot(summary_df, dname, hr_plot_path)

        med_sample = X_test.iloc[[len(X_test) // 2]]
        t_grid = np.linspace(y_test_time.min(), y_test_time.max(), 50)
        s_mean, s_low, s_up = bayes_model.predict_survival_with_credible_intervals(
            med_sample, t_grid
        )

        surv_plot_path = os.path.join(
            FIGURES_DIR, f"bayesian_cox_{dname}_credible_survival.png"
        )
        draw_bayesian_credible_survival_plot(
            t_grid, s_mean[0], s_low[0], s_up[0], dname, surv_plot_path
        )

        all_results[dname] = {
            "posterior_summary": summary_df.to_dict(orient="records"),
            "mcmc_diagnostics": diag_list,
            "test_eval": test_eval,
            "cv_results": cv_results,
            "comparative_benchmarks": {
                "cox_ph": cox_c,
                "rsf": rsf_c,
                "deepsurv": ds_c,
                "bayesian_cox": test_eval["c_index"],
            },
            "resolved_config": {
                "method": args.method,
                "prior": args.prior,
                "prior_params": prior_params,
                "draws": args.draws,
                "tune": args.tune,
                "chains": args.chains,
                "advi_iterations": args.advi_iterations,
                "n_intervals": args.intervals,
            },
        }

    # 6. Save JSON table artifact
    json_path = os.path.join(TABLES_DIR, "bayesian_cox_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"BAYESIAN COX EXECUTION COMPLETE! Results saved to {json_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
