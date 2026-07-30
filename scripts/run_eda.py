"""
Full EDA Master Script.
Executes Exploratory Data Analysis across GBSG2, WHAS500, and METABRIC raw datasets.
Generates statistical outputs, publication figures, tables, and comprehensive markdown documentation.
"""

import json
import os
import sys

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.eda import EDAAnalyzer
from src.visualization.eda_plots import EDAPlotter

RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
TABLES_DIR = os.path.join(PROJECT_ROOT, "reports", "tables")
REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "eda_report.md")

datasets_config = {
    "GBSG2": {
        "file": "gbsg2.csv",
        "time": "time",
        "event": "cens",
        "categorical": ["horTh", "menostat"],
        "stratify_cols": ["horTh", "menostat"],
    },
    "WHAS500": {
        "file": "whas500.csv",
        "time": "lenfol",
        "event": "fstat",
        "categorical": ["gender", "cvd", "afb", "sho", "chf"],
        "stratify_cols": ["gender", "chf", "cvd"],
    },
    "METABRIC": {
        "file": "metabric.csv",
        "time": "duration",
        "event": "event",
        "categorical": [
            "tumour_stage",
            "chemotherapy",
            "hormone_therapy",
            "PAM50Subtype",
        ],
        "stratify_cols": ["PAM50Subtype", "tumour_stage", "hormone_therapy"],
    },
}


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    plotter = EDAPlotter(FIGURES_DIR)
    results = {}

    print("=" * 60)
    print("STARTING PHASE 3 — DATASET UNDERSTANDING & EDA")
    print("=" * 60)

    for name, cfg in datasets_config.items():
        filepath = os.path.join(RAW_DATA_DIR, cfg["file"])
        print(f"\nProcessing Dataset: {name} ({cfg['file']})...")

        analyzer = EDAAnalyzer(
            filepath=filepath,
            dataset_name=name,
            time_col=cfg["time"],
            event_col=cfg["event"],
            categorical_cols=cfg["categorical"],
        )

        integrity = analyzer.dataset_integrity()
        feat_dict = analyzer.feature_dictionary()
        missing = analyzer.missing_value_analysis()
        duplicates = analyzer.duplicate_analysis()
        num_stats = analyzer.numerical_analysis()
        cat_stats = analyzer.categorical_analysis()
        survival_stats = analyzer.survival_target_analysis()
        km_overall = analyzer.kaplan_meier_analysis()

        km_stratified = {}
        for s_col in cfg["stratify_cols"]:
            km_stratified[s_col] = analyzer.kaplan_meier_analysis(s_col)

        correlations = analyzer.correlation_analysis()
        outliers = analyzer.outlier_detection()
        distributions = analyzer.feature_distribution_analysis()
        insights = analyzer.clinical_insights()

        results[name] = {
            "integrity": integrity,
            "feature_dictionary": feat_dict,
            "missing": missing,
            "duplicates": duplicates,
            "numerical_stats": num_stats,
            "categorical_stats": cat_stats,
            "survival_stats": survival_stats,
            "kaplan_meier_overall": km_overall,
            "kaplan_meier_stratified": km_stratified,
            "correlations": correlations,
            "outliers": outliers,
            "distributions": distributions,
            "clinical_insights": insights,
        }

        # --- Generate Visualizations ---
        # 1. Overall Kaplan-Meier
        plotter.plot_kaplan_meier(
            km_overall,
            f"{name} — Overall Kaplan-Meier Survival Curve",
            f"km_{name.lower()}_overall.png",
        )

        # 2. Stratified Kaplan-Meier
        for s_col, s_km in km_stratified.items():
            plotter.plot_kaplan_meier(
                s_km,
                f"{name} — Kaplan-Meier Stratified by {s_col}",
                f"km_{name.lower()}_{s_col.lower()}.png",
                strat_name=s_col,
            )

        # 3. Correlation Heatmap
        if correlations and "pearson" in correlations:
            plotter.plot_correlation_heatmap(
                correlations["pearson"],
                f"{name} — Pearson Correlation Matrix",
                f"corr_{name.lower()}_pearson.png",
            )

        # 4. Target Survival Time Histogram
        target_vals = [
            float(x) for x in analyzer.columns_data[cfg["time"]] if x is not None
        ]
        plotter.plot_histogram(
            target_vals,
            f"{name} — Survival Time Distribution ({cfg['time']})",
            f"Time ({cfg['time']})",
            f"hist_{name.lower()}_survival_time.png",
        )

        print(f"✓ Completed analysis & visualizations for {name}.")

    # Save JSON summary
    json_path = os.path.join(TABLES_DIR, "eda_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved structured EDA statistics to {json_path}")

    # Generate Markdown Report
    generate_markdown_report(results)
    print(f"Saved complete EDA markdown report to {REPORT_PATH}")
    print("\nPhase 3 EDA successfully executed!")


def generate_markdown_report(results):
    lines = []
    lines.append(
        "# Phase 3 — Exploratory Data Analysis & Dataset Understanding Report\n"
    )
    lines.append(
        "**Project**: Bayesian Cox Proportional Hazards & Deep Survival Models\n"
    )
    lines.append("**Execution Timestamp**: 2026-07-28\n")
    lines.append("---\n")

    for name, data in results.items():
        lines.append(f"## 1. {name} Dataset\n")

        # 1. Integrity
        integ = data["integrity"]
        lines.append("### Step 3.1 — Dataset Integrity\n")
        lines.append(f"- **Rows**: {integ['rows']}")
        lines.append(f"- **Columns**: {integ['cols']}")
        lines.append(f"- **Memory Usage**: {integ['memory_kb']} KB")
        lines.append(f"- **Time Column**: `{integ['time_col']}`")
        lines.append(f"- **Event Column**: `{integ['event_col']}`")
        lines.append(
            f"- **Duplicate Rows**: {integ['duplicate_rows']} ({integ['duplicate_pct']}%)\n"
        )

        # 2. Data Dictionary
        lines.append("### Step 3.2 — Data Dictionary\n")
        lines.append(
            "| Feature | Clinical Description | Data Type | Units / Range | Missing | Unique |"
        )
        lines.append(
            "| ------- | -------------------- | --------- | ------------- | ------- | ------ |"
        )
        for fd in data["feature_dictionary"]:
            lines.append(
                f"| `{fd['feature']}` | {fd['label']} — {fd['description']} | {fd['type']} | N/A | {fd['missing']} | {fd['unique_values']} |"
            )
        lines.append("\n")

        # 3. Missing & Duplicates
        lines.append("### Steps 3.3 & 3.4 — Missing Values & Duplicate Analysis\n")
        miss = data["missing"]
        lines.append(
            f"- **Total Missing Cells**: {miss['total_missing']} ({miss['total_missing_pct']}%)\n"
        )

        # 4. Numerical Analysis
        lines.append("### Step 3.5 — Numerical Feature Descriptive Statistics\n")
        lines.append(
            "| Feature | Mean | Median | Mode | Std Dev | Min | Max | Q1 | Q3 | IQR | Skewness | Kurtosis |"
        )
        lines.append(
            "| ------- | ---- | ------ | ---- | ------- | --- | --- | -- | -- | --- | -------- | -------- |"
        )
        for feat, ns in data["numerical_stats"].items():
            lines.append(
                f"| `{feat}` | {ns['mean']} | {ns['median']} | {ns['mode']} | {ns['std']} | {ns['min']} | {ns['max']} | {ns['q1']} | {ns['q3']} | {ns['iqr']} | {ns['skewness']} | {ns['kurtosis']} |"
            )
        lines.append("\n")

        # 5. Categorical Analysis
        lines.append("### Step 3.6 — Categorical Feature Analysis\n")
        for feat, cs in data["categorical_stats"].items():
            lines.append(
                f"#### Feature: `{feat}` (Imbalance Ratio: {cs['imbalance_ratio']}:1)"
            )
            for cat_k, cat_v in cs["frequencies"].items():
                lines.append(
                    f"  - `{cat_k}`: {cat_v['count']} records ({cat_v['percentage']}%)"
                )
        lines.append("\n")

        # 6. Survival Target Analysis
        lines.append("### Step 3.7 — Survival Target Statistics\n")
        surv = data["survival_stats"]
        lines.append(f"- **Total Patients**: {surv['total_patients']}")
        lines.append(f"- **Events Experienced (1)**: {surv['event_count']}")
        lines.append(f"- **Censored Patients (0)**: {surv['censored_count']}")
        lines.append(f"- **Censoring Rate**: **{surv['censoring_rate_pct']}%**")
        lines.append(
            f"- **Survival Time Range**: {surv['survival_time']['min']} to {surv['survival_time']['max']}"
        )
        lines.append(f"- **Mean Survival Time**: {surv['survival_time']['mean']}")
        lines.append(f"- **Median Survival Time**: {surv['survival_time']['median']}\n")

        # 7. Kaplan Meier Analysis
        lines.append("### Step 3.8 — Kaplan-Meier Survival Analysis\n")
        km_ov = data["kaplan_meier_overall"]
        lines.append(
            f"- **Overall Median Survival Time**: `{km_ov['median_survival_time']}`"
        )
        lines.append(
            f"- **Final Horizon Survival Probability**: `{km_ov['final_survival_prob']:.4f}`\n"
        )

        for strat_k, strat_v in data["kaplan_meier_stratified"].items():
            lines.append(f"#### Stratified by `{strat_k}`:")
            for grp_k, grp_v in strat_v.items():
                lines.append(
                    f"  - Group `{grp_k}`: Median Survival = `{grp_v['median_survival_time']}`, Final S(t) = `{grp_v['final_survival_prob']:.4f}`"
                )
            lines.append("")

        # 8. Outliers & Distribution
        lines.append(
            "### Steps 3.10 & 3.11 — Outlier Detection & Feature Distributions\n"
        )
        lines.append(
            "| Feature | IQR Outliers | Z-Score Outliers | Distribution Shape | Preprocessing Recommendation |"
        )
        lines.append(
            "| ------- | ------------ | ---------------- | ------------------ | ---------------------------- |"
        )
        for feat, out_v in data["outliers"].items():
            dist_v = data["distributions"].get(feat, {})
            lines.append(
                f"| `{feat}` | {out_v['iqr_outliers_count']} ({out_v['iqr_outliers_pct']}%) | {out_v['zscore_outliers_count']} ({out_v['zscore_outliers_pct']}%) | {dist_v.get('distribution_type', 'N/A')} | {dist_v.get('preprocessing_recommendation', 'N/A')} |"
            )
        lines.append("\n")

        # 9. Clinical Insights
        lines.append("### Step 3.12 — Clinical Insights & Observations\n")
        for q_k, q_ans in data["clinical_insights"].items():
            lines.append(f"- **{q_k.replace('_', ' ').title()}**: {q_ans}")
        lines.append("\n---\n")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
