"""
Data Validation Module for Bayesian Survival Models.
Performs strict schema validation, target label integrity checks, and data quality assertions
prior to preprocessing.
"""

from typing import Any


class DatasetValidator:
    """Validates raw survival datasets for structural and mathematical integrity."""

    def __init__(self, df, dataset_name: str, time_col: str, event_col: str):
        self.df = df
        self.dataset_name = dataset_name
        self.time_col = time_col
        self.event_col = event_col

    def validate(self) -> dict[str, Any]:
        """Runs comprehensive validation suite."""
        report = {
            "dataset_name": self.dataset_name,
            "passed": True,
            "errors": [],
            "warnings": [],
            "checks": {},
        }

        # 1. Schema check
        columns = list(self.df.columns)
        report["checks"]["total_rows"] = len(self.df)
        report["checks"]["total_cols"] = len(columns)

        if self.time_col not in columns:
            report["passed"] = False
            report["errors"].append(
                f"Target time column '{self.time_col}' missing from dataset."
            )

        if self.event_col not in columns:
            report["passed"] = False
            report["errors"].append(
                f"Target event column '{self.event_col}' missing from dataset."
            )

        if not report["passed"]:
            return report

        # 2. Target validation (time > 0)
        time_vals = self.df[self.time_col]
        invalid_time_count = (time_vals <= 0).sum()
        if invalid_time_count > 0:
            report["passed"] = False
            report["errors"].append(
                f"Found {invalid_time_count} non-positive (<=0) survival time values in '{self.time_col}'."
            )

        null_time_count = time_vals.isnull().sum()
        if null_time_count > 0:
            report["passed"] = False
            report["errors"].append(
                f"Found {null_time_count} missing values in target time column '{self.time_col}'."
            )

        # 3. Target validation (event binary 0/1)
        event_vals = self.df[self.event_col]
        unique_events = set(event_vals.dropna().unique())
        if not unique_events.issubset({0, 1}):
            report["passed"] = False
            report["errors"].append(
                f"Target event column '{self.event_col}' contains invalid non-binary values: {unique_events}."
            )

        null_event_count = event_vals.isnull().sum()
        if null_event_count > 0:
            report["passed"] = False
            report["errors"].append(
                f"Found {null_event_count} missing values in target event column '{self.event_col}'."
            )

        # 4. Check for completely null columns or infinite values
        for col in columns:
            if self.df[col].isnull().all():
                report["warnings"].append(f"Column '{col}' is entirely null.")

            # Check infinite if numeric
            if self.df[col].dtype in ["int64", "float64"]:
                import numpy as np

                inf_count = np.isinf(self.df[col]).sum()
                if inf_count > 0:
                    report["passed"] = False
                    report["errors"].append(
                        f"Column '{col}' contains {inf_count} infinite (inf) values."
                    )

        report["checks"]["valid_rows"] = len(self.df)
        report["checks"]["event_count"] = int(self.df[self.event_col].sum())
        report["checks"]["censored_count"] = int((self.df[self.event_col] == 0).sum())
        report["checks"]["censoring_rate"] = float(
            round((report["checks"]["censored_count"] / len(self.df)) * 100, 2)
        )

        return report
