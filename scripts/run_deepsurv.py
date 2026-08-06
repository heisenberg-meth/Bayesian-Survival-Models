"""
Master Execution Script for Phase 7 — DeepSurv (Deep Learning Survival Model).
Trains DeepSurv neural networks across GBSG2, WHAS500, and METABRIC preprocessed datasets.
Performs hyperparameter tuning on validation sets, evaluates test discrimination (C-index)
and calibration (IBS), computes calibration curves, permutation feature importances (VIMP),
executes 5-fold CV, compares with Cox PH and RSF baselines, and exports results and plots.
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

from src.evaluation.calibration import compute_calibration_curve
from src.evaluation.cross_validation import CrossValidationEvaluator
from src.evaluation.metrics import concordance_index, evaluate_survival_model
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


def draw_vimp_bar_chart(df_vimp, dataset_name, filepath):
    """Draws DeepSurv Permutation Feature Importance (VIMP) bar chart using Pillow."""
    img_w, img_h = 900, 600
    img = Image.new("RGB", (img_w, img_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    pad_left, pad_right, pad_top, pad_bot = 220, 50, 70, 60
    plot_w = img_w - pad_left - pad_right
    plot_h = img_h - pad_top - pad_bot

    draw.text(
        (img_w // 2 - 240, 20),
        f"DeepSurv Feature Importance (VIMP) — {dataset_name.upper()}",
        fill=(30, 41, 59),
    )
    draw.text(
        (img_w // 2 - 60, img_h - 35), "Drop in C-Index (VIMP)", fill=(51, 65, 85)
    )

    n_feats = len(df_vimp)
    row_h = plot_h / max(n_feats, 1)

    max_vimp = max(0.01, float(df_vimp["importance (VIMP)"].max()) * 1.2)

    for idx, row in df_vimp.iterrows():
        y_top = pad_top + int(idx * row_h) + 5
        y_bot = pad_top + int((idx + 1) * row_h) - 5
        feat_name = str(row["feature"])
        vimp_val = float(row["importance (VIMP)"])

        draw.text((20, y_top + 2), feat_name[:25], fill=(30, 41, 59))

        bar_w = int((vimp_val / max_vimp) * plot_w)
        draw.rectangle(
            [(pad_left, y_top), (pad_left + bar_w, y_bot)], fill=(147, 51, 234)
        )
        draw.text(
            (pad_left + bar_w + 10, y_top + 2), f"{vimp_val:.4f}", fill=(71, 85, 105)
        )

    img.save(filepath)


def draw_calibration_plot(mean_pred, obs_surv, eval_time, dataset_name, filepath):
    """Draws Calibration Curve at milestone time t using Pillow."""
    img_w, img_h = 600, 600
    img = Image.new("RGB", (img_w, img_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    pad_left, pad_right, pad_top, pad_bot = 80, 40, 70, 80
    plot_w = img_w - pad_left - pad_right
    plot_h = img_h - pad_top - pad_bot

    draw.text(
        (img_w // 2 - 200, 20),
        f"Calibration Curve (t={round(eval_time, 1)}) — {dataset_name.upper()}",
        fill=(30, 41, 59),
    )
    draw.text(
        (img_w // 2 - 120, img_h - 35), "Mean Predicted Survival", fill=(51, 65, 85)
    )
    draw.text((15, img_h // 2 - 60), "Observed Survival (KM)", fill=(51, 65, 85))

    # Y-axis line & X-axis line
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

    # Gridlines and ticks
    for val in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        # Y-ticks
        y_pos = img_h - pad_bot - int(val * plot_h)
        draw.line(
            [(pad_left, y_pos), (img_w - pad_right, y_pos)],
            fill=(226, 232, 240),
            width=1,
        )
        draw.text((pad_left - 45, y_pos - 8), f"{val:.1f}", fill=(71, 85, 105))

        # X-ticks
        x_pos = pad_left + int(val * plot_w)
        draw.line(
            [(x_pos, img_h - pad_bot), (x_pos, pad_top)], fill=(226, 232, 240), width=1
        )
        draw.text((x_pos - 15, img_h - pad_bot + 10), f"{val:.1f}", fill=(71, 85, 105))

    # Draw perfect calibration diagonal
    x0, y0 = pad_left, img_h - pad_bot
    x1, y1 = img_w - pad_right, pad_top
    draw.line([(x0, y0), (x1, y1)], fill=(100, 116, 139), width=2)  # perfect diagonal

    # Plot points and lines
    pts = []
    for p, o in zip(mean_pred, obs_surv):
        x_pos = pad_left + int(p * plot_w)
        y_pos = img_h - pad_bot - int(o * plot_h)
        pts.append((x_pos, y_pos))

    # Draw lines
    for i in range(len(pts) - 1):
        draw.line([pts[i], pts[i + 1]], fill=(147, 51, 234), width=3)

    # Draw point markers (circles)
    for pt in pts:
        r = 6
        draw.ellipse(
            [(pt[0] - r, pt[1] - r), (pt[0] + r, pt[1] + r)], fill=(147, 51, 234)
        )

    img.save(filepath)


def load_json_if_exists(path: str) -> dict:
    if not os.path.exists(path):
        return {}

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    print("=" * 60)
    print("STARTING PHASE 7 — DEEPSURV (DEEP LEARNING SURVIVAL)")
    print("=" * 60)

    # Load Cox PH and RSF results for comparative analysis
    cox_results_path = os.path.join(TABLES_DIR, "cox_ph_results.json")
    rsf_results_path = os.path.join(TABLES_DIR, "rsf_results.json")

    cox_results = load_json_if_exists(cox_results_path)
    rsf_results = load_json_if_exists(rsf_results_path)

    all_results = {}
    datasets = ["gbsg2", "whas500", "metabric"]

    for dname in datasets:
        print(f"\n[+] Executing DeepSurv Pipeline on Dataset: '{dname.upper()}'")
        dataset_dir = os.path.join(PROCESSED_DIR, dname)

        train_df = pd.read_csv(os.path.join(dataset_dir, "train.csv"))
        val_df = pd.read_csv(os.path.join(dataset_dir, "val.csv"))
        test_df = pd.read_csv(os.path.join(dataset_dir, "test.csv"))

        X_train = train_df.drop(columns=["time", "event"])
        y_train_time = train_df["time"].values
        y_train_event = train_df["event"].values

        X_val = val_df.drop(columns=["time", "event"])
        y_val_time = val_df["time"].values
        y_val_event = val_df["event"].values

        X_test = test_df.drop(columns=["time", "event"])
        y_test_time = test_df["time"].values
        y_test_event = test_df["event"].values

        # Hyperparameter tuning grid search
        param_grid = [
            {"hidden_dims": [16, 8], "l2_reg": 1e-4},
            {"hidden_dims": [16, 8], "l2_reg": 1e-3},
            {"hidden_dims": [32, 16], "l2_reg": 1e-4},
            {"hidden_dims": [32, 16], "l2_reg": 1e-3},
            {"hidden_dims": [32, 16], "l2_reg": 1e-2},
        ]

        print("    Running validation set grid search hyperparameter tuning...")
        best_val_c = -1.0
        best_params = None

        for p_idx, params in enumerate(param_grid):
            # Fit candidate model
            model = DeepSurvModel(
                hidden_dims=params["hidden_dims"],
                l2_reg=params["l2_reg"],
                random_state=42,
                max_iter=150,
            )
            model.fit(X_train, y_train_time, y_train_event)

            # Evaluate on validation set
            val_risk = model.predict_risk(X_val)
            val_c, _ = concordance_index(y_val_time, y_val_event, val_risk)
            print(f"      Combo {p_idx + 1}: {params} -> Val C-Index = {val_c:.4f}")

            if val_c > best_val_c:
                best_val_c = val_c
                best_params = params

        print(
            f"    [-->] Optimal Parameters: {best_params} with Val C-Index = {best_val_c:.4f}"
        )

        # Train final model with the optimal parameters
        deepsurv_model = DeepSurvModel(
            hidden_dims=best_params["hidden_dims"],
            l2_reg=best_params["l2_reg"],
            random_state=42,
            max_iter=300,
        )
        deepsurv_model.fit(X_train, y_train_time, y_train_event)

        arch_df = deepsurv_model.get_summary()
        print("\n    Neural Network Architecture Summary:")
        print(arch_df.to_string(index=False))

        # Permutation Feature Importance
        vimp_df = deepsurv_model.get_feature_importances()
        print("\n    Permutation Feature Importances (VIMP):")
        print(vimp_df.to_string(index=False))

        # Test Evaluation
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

        # 5-Fold Stratified Cross-Validation using the optimal hyperparameters
        def model_trainer(df_tr, df_v, best_params=best_params):
            X_tr = df_tr.drop(columns=["time", "event"])
            y_tr_t = df_tr["time"].values
            y_tr_e = df_tr["event"].values
            m = DeepSurvModel(
                hidden_dims=best_params["hidden_dims"],
                l2_reg=best_params["l2_reg"],
                random_state=42,
                max_iter=200,
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

        # Compare with Cox PH & RSF
        cox_c = cox_results.get(dname, {}).get("test_eval", {}).get("c_index", "N/A")
        rsf_c = rsf_results.get(dname, {}).get("test_eval", {}).get("c_index", "N/A")

        print("\n    Multi-Model Comparative Test C-Index:")
        print(f"    - Cox PH Baseline:         {cox_c}")
        print(f"    - Random Survival Forest:  {rsf_c}")
        print(f"    - DeepSurv Neural Net:     {test_eval['c_index']}")

        # Generate Figures
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

        # Permutation Feature Importance Bar Chart
        vimp_path = os.path.join(
            FIGURES_DIR, f"deepsurv_feature_importance_{dname}.png"
        )
        draw_vimp_bar_chart(vimp_df, dname, vimp_path)

        # Calibration Curve at median milestone time t
        median_time = eval_times[1]
        pred_surv_at_median = deepsurv_model.predict_survival(
            X_test, np.array([median_time])
        )[:, 0]
        mean_pred, obs_surv = compute_calibration_curve(
            y_test_time, y_test_event, pred_surv_at_median, median_time, n_bins=5
        )

        cal_path = os.path.join(FIGURES_DIR, f"deepsurv_{dname}_calibration.png")
        draw_calibration_plot(mean_pred, obs_surv, median_time, dname, cal_path)

        all_results[dname] = {
            "best_params": best_params,
            "architecture": arch_df.to_dict(orient="records"),
            "vimp": vimp_df.to_dict(orient="records"),
            "test_eval": test_eval,
            "cv_results": cv_results,
            "calibration_curve": {
                "mean_pred": mean_pred.tolist(),
                "observed_surv": obs_surv.tolist(),
                "eval_time": float(median_time),
            },
            "cox_baseline_test_c_index": cox_c,
            "rsf_test_c_index": rsf_c,
        }

    # Save JSON table artifact
    json_path = os.path.join(TABLES_DIR, "deepsurv_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"DEEPSURV EXECUTION COMPLETE! Results saved to {json_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
