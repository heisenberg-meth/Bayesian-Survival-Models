"""
Master Execution Script for Phase 6 — Cox Proportional Hazards Model (Baseline).
Trains Cox PH models across GBSG2, WHAS500, and METABRIC preprocessed datasets.
Evaluates test discrimination (C-index) and calibration (IBS), executes 5-fold CV,
performs Schoenfeld residuals assumption tests, exports results, and generates publication plots.

Hyperparameters are loaded from config/model.yaml; CLI arguments override when explicitly passed.
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

from src.evaluation.cross_validation import CrossValidationEvaluator
from src.evaluation.metrics import evaluate_survival_model
from src.models.cox import CoxPHModel
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
    """Load Cox PH config from config/model.yaml, with safe fallbacks."""
    try:
        cfg = get_project_config(os.path.join(PROJECT_ROOT, "config"))
        return cfg.models.cox_ph
    except Exception:  # noqa: BLE001
        from src.utils.config import ConfigDict

        return ConfigDict({"l2_reg": 0.0001})


def draw_baseline_survival_plot(eval_times, surv_curves, dataset_name, filepath):
    """Draws baseline survival curves using Pillow."""
    img_w, img_h = 900, 600
    img = Image.new("RGB", (img_w, img_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Padding
    pad_left, pad_right, pad_top, pad_bot = 80, 40, 70, 80
    plot_w = img_w - pad_left - pad_right
    plot_h = img_h - pad_top - pad_bot

    # Title & Axis Labels
    draw.text(
        (img_w // 2 - 180, 20),
        f"Cox PH Survival Curves — {dataset_name.upper()}",
        fill=(30, 41, 59),
    )
    draw.text((img_w // 2 - 50, img_h - 35), "Time Horizon", fill=(51, 65, 85))
    draw.text((15, img_h // 2 - 60), "Survival S(t)", fill=(51, 65, 85))

    # Axis Lines
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

    # Gridlines
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
        (37, 99, 235),
        (220, 38, 38),
        (16, 185, 129),
        (217, 119, 6),
        (147, 51, 234),
    ]

    # Draw curves
    for idx, (label, surv_vals) in enumerate(surv_curves.items()):
        color = colors[idx % len(colors)]
        pts = []
        for t, s in zip(eval_times, surv_vals):
            x_pos = pad_left + int((t / max_t) * plot_w)
            y_pos = img_h - pad_bot - int(s * plot_h)
            pts.append((x_pos, y_pos))

        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i + 1]], fill=color, width=3)

        # Legend
        leg_x = img_w - pad_right - 220
        leg_y = pad_top + 20 + idx * 25
        draw.rectangle([(leg_x, leg_y), (leg_x + 15, leg_y + 12)], fill=color)
        draw.text((leg_x + 25, leg_y - 2), label[:25], fill=(30, 41, 59))

    img.save(filepath)


def draw_forest_plot(df_summary, dataset_name, filepath):
    """Draws Hazard Ratio Forest Plot with 95% CIs using Pillow."""
    img_w, img_h = 900, 650
    img = Image.new("RGB", (img_w, img_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    pad_left, pad_right, pad_top, pad_bot = 220, 50, 70, 60
    plot_w = img_w - pad_left - pad_right
    plot_h = img_h - pad_top - pad_bot

    draw.text(
        (img_w // 2 - 200, 20),
        f"Cox PH Hazard Ratios (95% CI) — {dataset_name.upper()}",
        fill=(30, 41, 59),
    )
    draw.text(
        (img_w // 2 - 60, img_h - 35), "Hazard Ratio (log scale)", fill=(51, 65, 85)
    )

    n_feats = len(df_summary)
    row_h = plot_h / max(n_feats, 1)

    min_hr = max(0.1, float(df_summary["95% CI Lower"].min()) * 0.8)
    max_hr = min(10.0, float(df_summary["95% CI Upper"].max()) * 1.2)

    min_log = np.log(min_hr)
    max_log = np.log(max_hr)

    # Null line HR = 1.0 (log HR = 0)
    x_null = pad_left + int(((0.0 - min_log) / (max_log - min_log)) * plot_w)
    draw.line(
        [(x_null, pad_top), (x_null, img_h - pad_bot)], fill=(239, 68, 68), width=2
    )
    draw.text((x_null - 15, pad_top - 20), "HR = 1.0", fill=(239, 68, 68))

    for idx, row in df_summary.iterrows():
        y_center = pad_top + int((idx + 0.5) * row_h)
        feat_name = str(row["feature"])
        hr_val = float(row["exp(coef) HR"])
        ci_l = float(row["95% CI Lower"])
        ci_u = float(row["95% CI Upper"])

        # Label
        draw.text((20, y_center - 8), feat_name[:25], fill=(30, 41, 59))

        x_hr = pad_left + int(
            ((np.log(max(hr_val, 1e-4)) - min_log) / (max_log - min_log)) * plot_w
        )
        x_l = pad_left + int(
            ((np.log(max(ci_l, 1e-4)) - min_log) / (max_log - min_log)) * plot_w
        )
        x_u = pad_left + int(
            ((np.log(max(ci_u, 1e-4)) - min_log) / (max_log - min_log)) * plot_w
        )

        # CI line
        draw.line([(x_l, y_center), (x_u, y_center)], fill=(37, 99, 235), width=2)
        # End caps
        draw.line(
            [(x_l, y_center - 5), (x_l, y_center + 5)], fill=(37, 99, 235), width=2
        )
        draw.line(
            [(x_u, y_center - 5), (x_u, y_center + 5)], fill=(37, 99, 235), width=2
        )
        # Point estimate
        draw.ellipse(
            [(x_hr - 5, y_center - 5), (x_hr + 5, y_center + 5)], fill=(15, 23, 42)
        )

        # Text annotation
        draw.text(
            (img_w - 180, y_center - 8),
            f"{hr_val:.2f} [{ci_l:.2f}, {ci_u:.2f}]",
            fill=(71, 85, 105),
        )

    img.save(filepath)


def main():
    # -----------------------------------------------------------------------
    # Load canonical config from config/model.yaml
    # -----------------------------------------------------------------------
    cox_cfg = _load_model_config()
    cfg_l2_reg = cox_cfg.get("l2_reg", 0.0001)

    print("=" * 60)
    print("STARTING PHASE 6 — COX PROPORTIONAL HAZARDS MODEL (BASELINE)")
    print("=" * 60)
    print("  Resolved Configuration (source: config/model.yaml):")
    print(f"    L2 Regularization:   {cfg_l2_reg}")
    print("=" * 60)

    all_results = {}
    datasets = ["gbsg2", "whas500", "metabric"]

    for dname in datasets:
        print(f"\n[+] Executing Cox PH on Dataset: '{dname.upper()}'")
        dataset_dir = os.path.join(PROCESSED_DIR, dname)

        train_df = pd.read_csv(os.path.join(dataset_dir, "train.csv"))
        pd.read_csv(os.path.join(dataset_dir, "val.csv"))
        test_df = pd.read_csv(os.path.join(dataset_dir, "test.csv"))

        X_train = train_df.drop(columns=["time", "event", "subject_id"], errors="ignore")
        y_train_time = train_df["time"].values
        y_train_event = train_df["event"].values

        X_test = test_df.drop(columns=["time", "event", "subject_id"], errors="ignore")
        y_test_time = test_df["time"].values
        y_test_event = test_df["event"].values

        # 1. Fit Cox PH model — using config-sourced L2 regularization
        cox_model = CoxPHModel(l2_reg=cfg_l2_reg)
        cox_model.fit(X_train, y_train_time, y_train_event)

        summary_df = cox_model.get_summary()
        print("\n    Model Coefficients & Hazard Ratios:")
        print(summary_df.to_string(index=False))

        # 2. Check Proportional Hazards Assumption
        ph_df = cox_model.check_proportional_hazards(
            X_train, y_train_time, y_train_event
        )
        print("\n    Proportional Hazards Assumption Test (Schoenfeld Residuals):")
        print(ph_df.to_string(index=False))

        # 3. Test Evaluation
        eval_times = np.percentile(y_test_time, [25, 50, 75])
        test_risk = cox_model.predict_risk(X_test)

        def surv_fn(times, cox_model=cox_model, X_test=X_test):
            return cox_model.predict_survival(X_test, times)

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

        # 4. 5-Fold Stratified Cross-Validation
        def model_trainer(df_tr, df_v):
            X_tr = df_tr.drop(columns=["time", "event", "subject_id"], errors="ignore")
            y_tr_t = df_tr["time"].values
            y_tr_e = df_tr["event"].values
            m = CoxPHModel(l2_reg=cfg_l2_reg)
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

        # 5. Generate Figures
        forest_path = os.path.join(FIGURES_DIR, f"cox_ph_{dname}_hr_forest.png")
        draw_forest_plot(summary_df, dname, forest_path)

        surv_path = os.path.join(FIGURES_DIR, f"cox_ph_{dname}_baseline_survival.png")

        # Sample 3 synthetic risk profiles (Low, Medium, High risk)
        low_idx = np.argmin(test_risk)
        high_idx = np.argmax(test_risk)
        med_idx = np.argsort(test_risk)[len(test_risk) // 2]

        surv_curves = {
            "Low Risk Profile": surv_fn(eval_times)[low_idx],
            "Median Risk Profile": surv_fn(eval_times)[med_idx],
            "High Risk Profile": surv_fn(eval_times)[high_idx],
        }
        draw_baseline_survival_plot(eval_times, surv_curves, dname, surv_path)

        all_results[dname] = {
            "summary": summary_df.to_dict(orient="records"),
            "ph_assumption": ph_df.to_dict(orient="records"),
            "test_eval": test_eval,
            "cv_results": cv_results,
            "resolved_config": {"l2_reg": cfg_l2_reg},
        }

    # 6. Save JSON table artifact
    json_path = os.path.join(TABLES_DIR, "cox_ph_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"COX PH BASELINE EXECUTION COMPLETE! Results saved to {json_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
