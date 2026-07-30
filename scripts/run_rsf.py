"""
Master Execution Script for Phase 6 — Random Survival Forests (RSF).
Fits RSF models across GBSG2, WHAS500, and METABRIC preprocessed datasets.
Performs grid search hyperparameter tuning on validation sets, fits optimal ensembles,
evaluates test discrimination (C-index, time-dependent AUC) and calibration (IBS, calibration curve),
runs 5-fold CV, computes VIMP, exports results, and generates publication plots.
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
from src.evaluation.metrics import evaluate_survival_model
from src.models.random_survival_forest import RandomSurvivalForestModel

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
TABLES_DIR = os.path.join(REPORTS_DIR, "tables")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)


def draw_rsf_survival_plot(eval_times, surv_curves, dataset_name, filepath):
    """Draws RSF ensemble survival curves using Pillow."""
    img_w, img_h = 900, 600
    img = Image.new("RGB", (img_w, img_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    pad_left, pad_right, pad_top, pad_bot = 80, 40, 70, 80
    plot_w = img_w - pad_left - pad_right
    plot_h = img_h - pad_top - pad_bot

    draw.text(
        (img_w // 2 - 200, 20),
        f"Random Survival Forest Curves — {dataset_name.upper()}",
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
        (16, 185, 129),
        (37, 99, 235),
        (220, 38, 38),
        (217, 119, 6),
        (147, 51, 234),
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
    """Draws RSF Permutation Feature Importance (VIMP) bar chart."""
    img_w, img_h = 900, 600
    img = Image.new("RGB", (img_w, img_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    pad_left, pad_right, pad_top, pad_bot = 220, 50, 70, 60
    plot_w = img_w - pad_left - pad_right
    plot_h = img_h - pad_top - pad_bot

    draw.text(
        (img_w // 2 - 220, 20),
        f"RSF Feature Importance (VIMP) — {dataset_name.upper()}",
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
            [(pad_left, y_top), (pad_left + bar_w, y_bot)], fill=(16, 185, 129)
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

    # Draw perfect calibration diagonal (dashed/dotted or thin line)
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
        draw.line([pts[i], pts[i + 1]], fill=(37, 99, 235), width=3)

    # Draw point markers (circles)
    for pt in pts:
        r = 6
        draw.ellipse(
            [(pt[0] - r, pt[1] - r), (pt[0] + r, pt[1] + r)], fill=(37, 99, 235)
        )

    img.save(filepath)


def main():
    print("=" * 60)
    print("STARTING PHASE 6 — RANDOM SURVIVAL FORESTS (RSF)")
    print("=" * 60)

    # Load Cox PH baseline results for comparative analysis
    cox_results_path = os.path.join(TABLES_DIR, "cox_ph_results.json")
    cox_results = {}
    if os.path.exists(cox_results_path):
        with open(cox_results_path, "r", encoding="utf-8") as f:
            cox_results = json.load(f)

    all_results = {}
    datasets = ["gbsg2", "whas500", "metabric"]

    # Define hyperparameter grid for tuning
    param_grid = [
        {
            "n_estimators": 40,
            "max_depth": 4,
            "min_samples_leaf": 3,
            "min_samples_split": 6,
            "max_features": "sqrt",
            "bootstrap": True,
        },
        {
            "n_estimators": 40,
            "max_depth": 6,
            "min_samples_leaf": 3,
            "min_samples_split": 6,
            "max_features": "sqrt",
            "bootstrap": True,
        },
        {
            "n_estimators": 40,
            "max_depth": 6,
            "min_samples_leaf": 5,
            "min_samples_split": 10,
            "max_features": "sqrt",
            "bootstrap": True,
        },
        {
            "n_estimators": 75,
            "max_depth": 4,
            "min_samples_leaf": 3,
            "min_samples_split": 6,
            "max_features": "sqrt",
            "bootstrap": True,
        },
        {
            "n_estimators": 75,
            "max_depth": 6,
            "min_samples_leaf": 3,
            "min_samples_split": 6,
            "max_features": "sqrt",
            "bootstrap": True,
        },
        {
            "n_estimators": 75,
            "max_depth": 6,
            "min_samples_leaf": 5,
            "min_samples_split": 10,
            "max_features": "sqrt",
            "bootstrap": True,
        },
    ]

    for dname in datasets:
        print(f"\n[+] Processing Dataset: '{dname.upper()}'")
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

        print("    Running grid search hyperparameter tuning on validation set...")
        best_val_c = -1.0
        best_params = None

        for p_idx, params in enumerate(param_grid):
            # Fit candidate model
            model = RandomSurvivalForestModel(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                min_samples_leaf=params["min_samples_leaf"],
                max_features=params["max_features"],
                bootstrap=params["bootstrap"],
                random_state=42,
            )
            model.fit(X_train, y_train_time, y_train_event)

            # Evaluate on validation set
            val_risk = model.predict_risk(X_val)
            from src.evaluation.metrics import concordance_index

            val_c, _ = concordance_index(y_val_time, y_val_event, val_risk)

            print(f"      Combo {p_idx + 1}: {params} -> Val C-Index = {val_c:.4f}")

            if val_c > best_val_c:
                best_val_c = val_c
                best_params = params

        print(
            f"    [-->] Optimal Parameters: {best_params} with Val C-Index = {best_val_c:.4f}"
        )

        # Train final model with the optimal parameters
        final_model = RandomSurvivalForestModel(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            min_samples_split=best_params["min_samples_split"],
            min_samples_leaf=best_params["min_samples_leaf"],
            max_features=best_params["max_features"],
            bootstrap=best_params["bootstrap"],
            random_state=42,
        )
        final_model.fit(X_train, y_train_time, y_train_event)

        vimp_df = final_model.get_summary()
        print("\n    Permutation Feature Importances (VIMP):")
        print(vimp_df.to_string(index=False))

        # Test Evaluation
        eval_times = np.percentile(y_test_time, [25, 50, 75])
        test_risk = final_model.predict_risk(X_test)

        def surv_fn(times, final_model=final_model, X_test=X_test):
            return final_model.predict_survival(X_test, times)

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
        print(
            f"    Test Time-dependent AUC:   {test_eval.get('time_dependent_auc', 'N/A')}"
        )

        # Compute Calibration Curve at median milestone time t
        median_time = eval_times[1]  # 50th percentile
        pred_surv_at_median = final_model.predict_survival(
            X_test, np.array([median_time])
        )[:, 0]
        mean_pred, obs_surv = compute_calibration_curve(
            y_test_time, y_test_event, pred_surv_at_median, median_time, n_bins=5
        )

        # 5-Fold Stratified Cross-Validation (on training folds)
        def model_trainer(df_tr, df_v, best_params=best_params):
            X_tr = df_tr.drop(columns=["time", "event"])
            y_tr_t = df_tr["time"].values
            y_tr_e = df_tr["event"].values
            m = RandomSurvivalForestModel(
                n_estimators=best_params["n_estimators"],
                max_depth=best_params["max_depth"],
                min_samples_split=best_params["min_samples_split"],
                min_samples_leaf=best_params["min_samples_leaf"],
                max_features=best_params["max_features"],
                bootstrap=best_params["bootstrap"],
                random_state=42,
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

        # Compare with Cox PH Baseline
        cox_c = cox_results.get(dname, {}).get("test_eval", {}).get("c_index", "N/A")
        cox_ibs = (
            cox_results.get(dname, {})
            .get("test_eval", {})
            .get("integrated_brier_score", "N/A")
        )
        print("\n    Baseline Comparison vs Cox PH:")
        print(f"    - Cox PH Test C-Index:     {cox_c}")
        print(f"    - RSF Test C-Index:        {test_eval['c_index']}")
        print(f"    - Cox PH Test IBS:         {cox_ibs}")
        print(
            f"    - RSF Test IBS:            {test_eval.get('integrated_brier_score', 'N/A')}"
        )

        # Generate Figures
        vimp_path = os.path.join(FIGURES_DIR, f"rsf_feature_importance_{dname}.png")
        draw_vimp_bar_chart(vimp_df, dname, vimp_path)

        surv_path = os.path.join(FIGURES_DIR, f"rsf_{dname}_survival_curves.png")
        low_idx = np.argmin(test_risk)
        high_idx = np.argmax(test_risk)
        med_idx = np.argsort(test_risk)[len(test_risk) // 2]

        surv_curves = {
            "Low Risk Profile": surv_fn(eval_times)[low_idx],
            "Median Risk Profile": surv_fn(eval_times)[med_idx],
            "High Risk Profile": surv_fn(eval_times)[high_idx],
        }
        draw_rsf_survival_plot(eval_times, surv_curves, dname, surv_path)

        cal_path = os.path.join(FIGURES_DIR, f"rsf_{dname}_calibration.png")
        draw_calibration_plot(mean_pred, obs_surv, median_time, dname, cal_path)

        all_results[dname] = {
            "best_params": best_params,
            "vimp": vimp_df.to_dict(orient="records"),
            "test_eval": test_eval,
            "cv_results": cv_results,
            "calibration_curve": {
                "mean_pred": mean_pred.tolist(),
                "observed_surv": obs_surv.tolist(),
                "eval_time": float(median_time),
            },
            "cox_baseline_test_c_index": cox_c,
            "cox_baseline_test_ibs": cox_ibs,
        }

    # Save JSON table artifact
    json_path = os.path.join(TABLES_DIR, "rsf_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"RANDOM SURVIVAL FOREST EXECUTION COMPLETE! Results saved to {json_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
