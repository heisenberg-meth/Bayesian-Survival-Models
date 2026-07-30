"""
Master Execution Script for Phase 8 — DeepSurv (Deep Learning Survival Model).
Trains DeepSurv neural networks across GBSG2, WHAS500, and METABRIC preprocessed datasets.
Evaluates test discrimination (C-index) and calibration (IBS), executes 5-fold CV,
compares with Cox PH and RSF baselines, exports results, and generates publication plots.
"""

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
from src.models.deepsurv import DeepSurvModel

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
TABLES_DIR = os.path.join(REPORTS_DIR, "tables")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)


def draw_training_curve_plot(loss_history, dataset_name, filepath):
    """Draws DeepSurv neural network training loss curve using Pillow."""
    img_w, img_h = 900, 600
    img = Image.new("RGB", (img_w, img_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    pad_left, pad_right, pad_top, pad_bot = 90, 40, 70, 80
    plot_w = img_w - pad_left - pad_right
    plot_h = img_h - pad_top - pad_bot

    draw.text(
        (img_w // 2 - 220, 20),
        f"DeepSurv Training Loss Curve — {dataset_name.upper()}",
        fill=(30, 41, 59),
    )
    draw.text(
        (img_w // 2 - 50, img_h - 35), "Optimization Iteration", fill=(51, 65, 85)
    )
    draw.text((15, img_h // 2 - 60), "Loss (Partial Likelihood)", fill=(51, 65, 85))

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

    n_iters = len(loss_history)
    min_l = min(loss_history)
    max_l = max(loss_history)
    l_range = max(1e-5, max_l - min_l)

    # Gridlines
    for i in range(5):
        frac = i / 4.0
        val = min_l + frac * l_range
        y_pos = img_h - pad_bot - int(frac * plot_h)
        draw.line(
            [(pad_left, y_pos), (img_w - pad_right, y_pos)],
            fill=(226, 232, 240),
            width=1,
        )
        draw.text((10, y_pos - 8), f"{val:.1f}", fill=(71, 85, 105))

    pts = []
    for idx, l_val in enumerate(loss_history):
        x_pos = pad_left + int((idx / max(1, n_iters - 1)) * plot_w)
        norm_y = (l_val - min_l) / l_range
        y_pos = img_h - pad_bot - int(norm_y * plot_h)
        pts.append((x_pos, y_pos))

    for i in range(len(pts) - 1):
        draw.line([pts[i], pts[i + 1]], fill=(147, 51, 234), width=3)

    img.save(filepath)


def draw_deepsurv_survival_plot(eval_times, surv_curves, dataset_name, filepath):
    """Draws DeepSurv survival curves using Pillow."""
    img_w, img_h = 900, 600
    img = Image.new("RGB", (img_w, img_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    pad_left, pad_right, pad_top, pad_bot = 80, 40, 70, 80
    plot_w = img_w - pad_left - pad_right
    plot_h = img_h - pad_top - pad_bot

    draw.text(
        (img_w // 2 - 200, 20),
        f"DeepSurv Neural Curves — {dataset_name.upper()}",
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
    colors = [
        (147, 51, 234),
        (37, 99, 235),
        (220, 38, 38),
        (16, 185, 129),
        (217, 119, 6),
    ]

    for idx, (label, surv_vals) in enumerate(surv_curves.items()):
        color = colors[idx % len(colors)]
        pts = []
        for t, s in zip(eval_times, surv_vals):
            x_pos = pad_left + int((t / max_t) * plot_w)
            y_pos = img_h - pad_bot - int(s * plot_h)
            pts.append((x_pos, y_pos))

        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i + 1]], fill=color, width=3)

        leg_x = img_w - pad_right - 220
        leg_y = pad_top + 20 + idx * 25
        draw.rectangle([(leg_x, leg_y), (leg_x + 15, leg_y + 12)], fill=color)
        draw.text((leg_x + 25, leg_y - 2), label[:25], fill=(30, 41, 59))

    img.save(filepath)


def main():
    print("=" * 60)
    print("STARTING PHASE 8 — DEEPSURV (DEEP LEARNING SURVIVAL)")
    print("=" * 60)

    # Load Cox PH and RSF results for comparative analysis
    cox_results_path = os.path.join(TABLES_DIR, "cox_ph_results.json")
    rsf_results_path = os.path.join(TABLES_DIR, "rsf_results.json")

    cox_results = (
        json.load(open(cox_results_path, "r"))
        if os.path.exists(cox_results_path)
        else {}
    )
    rsf_results = (
        json.load(open(rsf_results_path, "r"))
        if os.path.exists(rsf_results_path)
        else {}
    )

    all_results = {}
    datasets = ["gbsg2", "whas500", "metabric"]

    for dname in datasets:
        print(f"\n[+] Training DeepSurv Neural Network on Dataset: '{dname.upper()}'")
        dataset_dir = os.path.join(PROCESSED_DIR, dname)

        train_df = pd.read_csv(os.path.join(dataset_dir, "train.csv"))
        pd.read_csv(os.path.join(dataset_dir, "val.csv"))
        test_df = pd.read_csv(os.path.join(dataset_dir, "test.csv"))

        X_train = train_df.drop(columns=["time", "event"])
        y_train_time = train_df["time"].values
        y_train_event = train_df["event"].values

        X_test = test_df.drop(columns=["time", "event"])
        y_test_time = test_df["time"].values
        y_test_event = test_df["event"].values

        # 1. Fit DeepSurv Model
        deepsurv_model = DeepSurvModel(
            hidden_dims=[32, 16], l2_reg=1e-3, random_state=42, max_iter=300
        )
        deepsurv_model.fit(X_train, y_train_time, y_train_event)

        arch_df = deepsurv_model.get_summary()
        print("\n    Neural Network Architecture:")
        print(arch_df.to_string(index=False))

        # 2. Test Evaluation
        eval_times = np.percentile(y_test_time, [25, 50, 75])
        test_risk = deepsurv_model.predict_risk(X_test)

        def surv_fn(times, deepsurv_model=deepsurv_model, X_test=X_test):
            return deepsurv_model.predict_survival(X_test, times)

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
            m = DeepSurvModel(
                hidden_dims=[32, 16], l2_reg=1e-3, random_state=42, max_iter=200
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

        # 4. Compare with Cox PH & RSF
        cox_c = cox_results.get(dname, {}).get("test_eval", {}).get("c_index", "N/A")
        rsf_c = rsf_results.get(dname, {}).get("test_eval", {}).get("c_index", "N/A")

        print("\n    Multi-Model Comparative Test C-Index:")
        print(f"    - Cox PH Baseline:         {cox_c}")
        print(f"    - Random Survival Forest:  {rsf_c}")
        print(f"    - DeepSurv Neural Net:     {test_eval['c_index']}")

        # 5. Generate Figures
        loss_curve_path = os.path.join(
            FIGURES_DIR, f"deepsurv_{dname}_training_curves.png"
        )
        draw_training_curve_plot(
            deepsurv_model.training_loss_history, dname, loss_curve_path
        )

        surv_path = os.path.join(FIGURES_DIR, f"deepsurv_{dname}_survival_curves.png")
        low_idx = np.argmin(test_risk)
        high_idx = np.argmax(test_risk)
        med_idx = np.argsort(test_risk)[len(test_risk) // 2]

        surv_curves = {
            "Low Risk Profile": surv_fn(eval_times)[low_idx],
            "Median Risk Profile": surv_fn(eval_times)[med_idx],
            "High Risk Profile": surv_fn(eval_times)[high_idx],
        }
        draw_deepsurv_survival_plot(eval_times, surv_curves, dname, surv_path)

        all_results[dname] = {
            "architecture": arch_df.to_dict(orient="records"),
            "test_eval": test_eval,
            "cv_results": cv_results,
            "cox_baseline_test_c_index": cox_c,
            "rsf_test_c_index": rsf_c,
        }

    # 6. Save JSON table artifact
    json_path = os.path.join(TABLES_DIR, "deepsurv_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"DEEPSURV EXECUTION COMPLETE! Results saved to {json_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
