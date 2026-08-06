"""
Execution Script for Phase 12 — Advanced Research Extensions.
Runs Ablation Studies of Priors, Censoring Robustness, and Small Dataset Analysis.
Exports results as JSON tables and generates publication-quality line charts.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.metrics import evaluate_survival_model
from src.models.bayesian.model import BayesianCoxModel
from src.models.cox import CoxPHModel
from src.models.deepsurv import DeepSurvModel
from src.models.random_survival_forest import RandomSurvivalForestModel

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
TABLES_DIR = os.path.join(REPORTS_DIR, "tables")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)


def load_gbsg2_data():
    dataset_dir = os.path.join(PROCESSED_DIR, "gbsg2")
    train_df = pd.read_csv(os.path.join(dataset_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(dataset_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(dataset_dir, "test.csv"))
    return train_df, val_df, test_df


def simulate_censoring(df, target_censoring_rate):
    df_new = df.copy()
    current_censored = np.sum(df_new["event"] == 0)
    total = len(df_new)
    target_censored = int(total * target_censoring_rate)
    to_censor = target_censored - current_censored

    if to_censor > 0:
        event_indices = df_new[df_new["event"] == 1].index.values
        rng = np.random.RandomState(42)
        censor_idx = rng.choice(
            event_indices, size=min(to_censor, len(event_indices)), replace=False
        )
        df_new.loc[censor_idx, "event"] = 0
    return df_new


def draw_line_plot(data, x_key, title, filename, x_label, y_label):
    img_w, img_h = 900, 600
    img = Image.new("RGB", (img_w, img_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    pad_left, pad_right, pad_top, pad_bot = 100, 200, 80, 80
    plot_w = img_w - pad_left - pad_right
    plot_h = img_h - pad_top - pad_bot

    # Title
    draw.text((img_w // 2 - 180, 20), title, fill=(30, 41, 59))

    # Draw axes
    draw.line(
        [(pad_left, img_h - pad_bot), (img_w - pad_right, img_h - pad_bot)],
        fill=(71, 85, 105),
        width=2,
    )
    draw.line(
        [(pad_left, pad_top), (pad_left, img_h - pad_bot)],
        fill=(71, 85, 105),
        width=2,
    )

    # Grid lines & ticks
    # Y-axis bounds: C-index from 0.35 to 0.70
    y_min, y_max = 0.35, 0.70
    for tick in np.linspace(y_min, y_max, 8):
        y_pos = img_h - pad_bot - int((tick - y_min) / (y_max - y_min) * plot_h)
        draw.line(
            [(pad_left, y_pos), (img_w - pad_right, y_pos)],
            fill=(226, 232, 240),
            width=1,
        )
        draw.text((pad_left - 45, y_pos - 5), f"{tick:.2f}", fill=(71, 85, 105))

    # X-axis ticks
    x_vals = [d[x_key] for d in data]
    x_min, x_max = min(x_vals), max(x_vals)
    for x_val in x_vals:
        x_pct = 0.5 if x_max == x_min else (x_val - x_min) / (x_max - x_min)
        x_pos = pad_left + int(x_pct * plot_w)
        draw.line(
            [(x_pos, img_h - pad_bot), (x_pos, img_h - pad_bot + 5)],
            fill=(71, 85, 105),
            width=2,
        )
        if x_key == "censoring_rate" or x_key == "proportion":
            draw.text(
                (x_pos - 15, img_h - pad_bot + 10),
                f"{x_val * 100:.0f}%",
                fill=(71, 85, 105),
            )
        else:
            draw.text(
                (x_pos - 15, img_h - pad_bot + 10), f"{x_val}", fill=(71, 85, 105)
            )

    # Labels
    draw.text((img_w // 2 - 50, img_h - 40), x_label, fill=(30, 41, 59))

    # We have 4 models: Cox PH, RSF, DeepSurv, Bayesian Cox
    models = ["Cox PH", "RSF", "DeepSurv", "Bayesian Cox"]
    colors = {
        "Cox PH": (59, 130, 246),  # Blue
        "RSF": (16, 185, 129),  # Green
        "DeepSurv": (147, 51, 234),  # Purple
        "Bayesian Cox": (249, 115, 22),  # Orange
    }

    # Draw curves
    for model_name in models:
        pts = []
        for d in data:
            x_val = d[x_key]
            x_pct = 0.5 if x_max == x_min else (x_val - x_min) / (x_max - x_min)
            x_pos = pad_left + int(x_pct * plot_w)

            y_val = d[model_name]
            y_pct = (y_val - y_min) / (y_max - y_min)
            y_pos = img_h - pad_bot - int(y_pct * plot_h)
            pts.append((x_pos, y_pos))

        color = colors[model_name]
        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i + 1]], fill=color, width=3)

        # Draw points
        for pt in pts:
            r = 5
            draw.ellipse([(pt[0] - r, pt[1] - r), (pt[0] + r, pt[1] + r)], fill=color)

    # Draw legend
    leg_x = img_w - pad_right + 20
    leg_y = pad_top
    for idx, model_name in enumerate(models):
        y_pos = leg_y + idx * 30
        color = colors[model_name]
        draw.rectangle([(leg_x, y_pos), (leg_x + 15, y_pos + 15)], fill=color)
        draw.text((leg_x + 25, y_pos + 2), model_name, fill=(30, 41, 59))

    filepath = os.path.join(FIGURES_DIR, filename)
    img.save(filepath)
    print(f"  [+] Saved line plot to: {filepath}")


def run_ablation_study(train_df, test_df):
    print("\n" + "=" * 60)
    print("RUNNING STEP 12.1 — ABLATION STUDY (PRIOR DISTRIBUTIONS)")
    print("=" * 60)

    X_train = train_df.drop(columns=["time", "event"])
    y_train_time = train_df["time"].values
    y_train_event = train_df["event"].values

    X_test = test_df.drop(columns=["time", "event"])
    y_test_time = test_df["time"].values
    y_test_event = test_df["event"].values

    eval_times = np.percentile(y_test_time, [25, 50, 75])

    priors = ["normal", "student-t", "laplace"]
    results = []

    for prior in priors:
        print(f"  Training Bayesian Cox Model with '{prior}' prior...")
        model = BayesianCoxModel(
            n_intervals=6,
            inference_method="advi",
            n_advi_iterations=1200,
            draws=400,
            random_state=42,
            coefficient_prior=prior,
        )
        model.fit(X_train, y_train_time, y_train_event)

        test_risk = model.predict_risk(X_test)

        def surv_fn(times, m=model, X=X_test):
            return m.predict_survival(X, times)

        eval_res = evaluate_survival_model(
            y_time=y_test_time,
            y_event=y_test_event,
            risk_scores=test_risk,
            surv_prob_fn=surv_fn,
            eval_times=eval_times,
        )

        results.append(
            {
                "prior": prior,
                "c_index": eval_res["c_index"],
                "c_index_se": eval_res["c_index_se"],
                "integrated_brier_score": eval_res.get(
                    "integrated_brier_score", np.nan
                ),
            }
        )
        print(
            f"    -> C-index: {eval_res['c_index']:.4f}, IBS: {eval_res.get('integrated_brier_score', np.nan):.4f}"
        )

    # Save results
    with open(os.path.join(TABLES_DIR, "extension_ablation_study.json"), "w") as f:
        json.dump(results, f, indent=4)

    return results


def run_censoring_robustness(train_df, test_df):
    print("\n" + "=" * 60)
    print("RUNNING STEP 12.4 — CENSORING ROBUSTNESS ANALYSIS")
    print("=" * 60)

    rates = [0.60, 0.75, 0.90]
    results = []

    # Pre-split test set
    X_test = test_df.drop(columns=["time", "event"])
    y_test_time = test_df["time"].values
    y_test_event = test_df["event"].values
    eval_times = np.percentile(y_test_time, [25, 50, 75])

    for rate in rates:
        print(
            f"  Simulating training set with target censoring rate = {rate * 100:.1f}%..."
        )
        df_cens = simulate_censoring(train_df, rate)

        X_tr = df_cens.drop(columns=["time", "event"])
        y_tr_time = df_cens["time"].values
        y_tr_event = df_cens["event"].values

        # 1. Cox PH
        print("    Training Cox PH...")
        cox_model = CoxPHModel(l2_reg=0.0)
        cox_model.fit(X_tr, y_tr_time, y_tr_event)
        cox_risk = cox_model.predict_risk(X_test)

        def cox_surv(t, m=cox_model, X=X_test):
            return m.predict_survival(X, t)

        cox_eval = evaluate_survival_model(
            y_test_time, y_test_event, cox_risk, cox_surv, eval_times
        )

        # 2. RSF
        print("    Training RSF...")
        rsf_model = RandomSurvivalForestModel(
            n_estimators=50, max_depth=6, random_state=42
        )
        rsf_model.fit(X_tr, y_tr_time, y_tr_event)
        rsf_risk = rsf_model.predict_risk(X_test)

        def rsf_surv(t, m=rsf_model, X=X_test):
            return m.predict_survival(X, t)

        rsf_eval = evaluate_survival_model(
            y_test_time, y_test_event, rsf_risk, rsf_surv, eval_times
        )

        # 3. DeepSurv
        print("    Training DeepSurv...")
        ds_model = DeepSurvModel(
            hidden_dims=[16, 8], l2_reg=1e-3, random_state=42, max_iter=100
        )
        ds_model.fit(X_tr, y_tr_time, y_tr_event)
        ds_risk = ds_model.predict_risk(X_test)

        def ds_surv(t, m=ds_model, X=X_test):
            return m.predict_survival(X, t)

        ds_eval = evaluate_survival_model(
            y_test_time, y_test_event, ds_risk, ds_surv, eval_times
        )

        # 4. Bayesian Cox
        print("    Training Bayesian Cox...")
        bay_model = BayesianCoxModel(
            n_intervals=6, n_advi_iterations=1000, draws=300, random_state=42
        )
        bay_model.fit(X_tr, y_tr_time, y_tr_event)
        bay_risk = bay_model.predict_risk(X_test)

        def bay_surv(t, m=bay_model, X=X_test):
            return m.predict_survival(X, t)

        bay_eval = evaluate_survival_model(
            y_test_time, y_test_event, bay_risk, bay_surv, eval_times
        )

        results.append(
            {
                "censoring_rate": rate,
                "Cox PH": cox_eval["c_index"],
                "RSF": rsf_eval["c_index"],
                "DeepSurv": ds_eval["c_index"],
                "Bayesian Cox": bay_eval["c_index"],
            }
        )
        print(
            f"    -> Results (C-index) at {rate * 100:.1f}% Censoring: Cox PH={cox_eval['c_index']:.4f}, RSF={rsf_eval['c_index']:.4f}, DeepSurv={ds_eval['c_index']:.4f}, Bayesian Cox={bay_eval['c_index']:.4f}"
        )

    with open(
        os.path.join(TABLES_DIR, "extension_censoring_robustness.json"), "w"
    ) as f:
        json.dump(results, f, indent=4)

    return results


def run_small_dataset_analysis(train_df, test_df):
    print("\n" + "=" * 60)
    print("RUNNING STEP 12.5 — SMALL DATASET ROBUSTNESS ANALYSIS")
    print("=" * 60)

    proportions = [0.20, 0.40, 0.60, 0.80, 1.00]
    results = []

    # Pre-split test set
    X_test = test_df.drop(columns=["time", "event"])
    y_test_time = test_df["time"].values
    y_test_event = test_df["event"].values
    eval_times = np.percentile(y_test_time, [25, 50, 75])

    for prop in proportions:
        print(f"  Training with {prop * 100:.1f}% of GBSG2 training dataset...")
        if prop < 1.0:
            df_sub = train_df.sample(frac=prop, random_state=42).copy()
        else:
            df_sub = train_df.copy()

        X_tr = df_sub.drop(columns=["time", "event"])
        y_tr_time = df_sub["time"].values
        y_tr_event = df_sub["event"].values

        # 1. Cox PH
        cox_model = CoxPHModel(l2_reg=0.0)
        cox_model.fit(X_tr, y_tr_time, y_tr_event)
        cox_risk = cox_model.predict_risk(X_test)

        def cox_surv(t, m=cox_model, X=X_test):
            return m.predict_survival(X, t)

        cox_eval = evaluate_survival_model(
            y_test_time, y_test_event, cox_risk, cox_surv, eval_times
        )

        # 2. RSF
        rsf_model = RandomSurvivalForestModel(
            n_estimators=50, max_depth=6, random_state=42
        )
        rsf_model.fit(X_tr, y_tr_time, y_tr_event)
        rsf_risk = rsf_model.predict_risk(X_test)

        def rsf_surv(t, m=rsf_model, X=X_test):
            return m.predict_survival(X, t)

        rsf_eval = evaluate_survival_model(
            y_test_time, y_test_event, rsf_risk, rsf_surv, eval_times
        )

        # 3. DeepSurv
        ds_model = DeepSurvModel(
            hidden_dims=[16, 8], l2_reg=1e-3, random_state=42, max_iter=100
        )
        ds_model.fit(X_tr, y_tr_time, y_tr_event)
        ds_risk = ds_model.predict_risk(X_test)

        def ds_surv(t, m=ds_model, X=X_test):
            return m.predict_survival(X, t)

        ds_eval = evaluate_survival_model(
            y_test_time, y_test_event, ds_risk, ds_surv, eval_times
        )

        # 4. Bayesian Cox
        bay_model = BayesianCoxModel(
            n_intervals=6, n_advi_iterations=1000, draws=300, random_state=42
        )
        bay_model.fit(X_tr, y_tr_time, y_tr_event)
        bay_risk = bay_model.predict_risk(X_test)

        def bay_surv(t, m=bay_model, X=X_test):
            return m.predict_survival(X, t)

        bay_eval = evaluate_survival_model(
            y_test_time, y_test_event, bay_risk, bay_surv, eval_times
        )

        results.append(
            {
                "proportion": prop,
                "Cox PH": cox_eval["c_index"],
                "RSF": rsf_eval["c_index"],
                "DeepSurv": ds_eval["c_index"],
                "Bayesian Cox": bay_eval["c_index"],
            }
        )
        print(
            f"    -> Results (C-index) at {prop * 100:.1f}% training size: Cox PH={cox_eval['c_index']:.4f}, RSF={rsf_eval['c_index']:.4f}, DeepSurv={ds_eval['c_index']:.4f}, Bayesian Cox={bay_eval['c_index']:.4f}"
        )

    with open(
        os.path.join(TABLES_DIR, "extension_small_dataset_analysis.json"), "w"
    ) as f:
        json.dump(results, f, indent=4)

    return results


def main():
    print("=" * 60)
    print("STARTING PHASE 12 — ADVANCED RESEARCH EXTENSIONS")
    print("=" * 60)

    train_df, _, test_df = load_gbsg2_data()

    # Step 12.1: Ablation Study
    run_ablation_study(train_df, test_df)

    # Step 12.4: Censoring Robustness
    cens_res = run_censoring_robustness(train_df, test_df)
    draw_line_plot(
        cens_res,
        "censoring_rate",
        "Model Discrimination vs. Censoring Rate (GBSG2)",
        "extension_censoring_robustness.png",
        "Censoring Rate",
        "C-Index",
    )

    # Step 12.5: Small Dataset Analysis
    small_res = run_small_dataset_analysis(train_df, test_df)
    draw_line_plot(
        small_res,
        "proportion",
        "Model Discrimination vs. Training Dataset Size (GBSG2)",
        "extension_small_dataset_analysis.png",
        "Training Size Proportion",
        "C-Index",
    )

    print("\n" + "=" * 60)
    print("PHASE 12 EXTENSIONS COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
