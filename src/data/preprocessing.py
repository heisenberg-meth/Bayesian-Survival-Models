"""
Reusable Preprocessing Pipeline for Bayesian Survival Models.
Orchestrates end-to-end data validation, cleaning, imputation, categorical encoding,
feature scaling, feature engineering, stratified train/val/test splitting, and fold generation.
Prevents data leakage by fitting transformers strictly on the training set.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

from src.data.validators import DatasetValidator
from src.data.feature_engineering import FeatureEngineer
from src.data.split import StratifiedSurvivalSplitter

class SurvivalDataPipeline:
    """Master reusable preprocessing pipeline for survival datasets."""

    def __init__(
        self,
        dataset_name: str,
        time_col: str,
        event_col: str,
        categorical_cols: List[str],
        ordinal_cols: Optional[Dict[str, List[Any]]] = None,
        random_state: int = 42
    ):
        self.dataset_name = dataset_name.lower()
        self.time_col = time_col
        self.event_col = event_col
        self.categorical_cols = categorical_cols or []
        self.ordinal_cols = ordinal_cols or {}
        self.random_state = random_state

        self.feature_engineer = FeatureEngineer(self.dataset_name)
        self.splitter = StratifiedSurvivalSplitter(random_state=self.random_state)

        # Preprocessing state parameters (fitted on Train set)
        self.imputation_values = {}
        self.scaling_params = {}
        self.encoding_maps = {}
        self.feature_names = []

    def run(self, raw_df: pd.DataFrame, output_dir: str) -> Dict[str, Any]:
        """Runs the complete preprocessing workflow."""

        # Step 4.1: Data Validation
        validator = DatasetValidator(raw_df, self.dataset_name, self.time_col, self.event_col)
        val_report = validator.validate()

        if not val_report["passed"]:
            raise ValueError(f"Dataset validation failed for {self.dataset_name}: {val_report['errors']}")

        # Step 4.2: Data Cleaning & Column Standardization
        clean_df = raw_df.copy()
        clean_df = clean_df.drop_duplicates().reset_index(drop=True)

        # Standardize target names to 'time' and 'event'
        rename_map = {self.time_col: "time", self.event_col: "event"}
        clean_df = clean_df.rename(columns=rename_map)

        time_target = "time"
        event_target = "event"

        # Step 4.6: Feature Engineering
        engineered_df = self.feature_engineer.transform(clean_df)

        # Step 4.10: Stratified Train / Validation / Test Split (70 / 15 / 15)
        train_raw, val_raw, test_raw = self.splitter.train_val_test_split(
            engineered_df,
            event_col=event_target,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15
        )

        # Separate targets from features
        X_train_raw = train_raw.drop(columns=[time_target, event_target])
        y_train_time = train_raw[time_target]
        y_train_event = train_raw[event_target]

        X_val_raw = val_raw.drop(columns=[time_target, event_target])
        y_val_time = val_raw[time_target]
        y_val_event = val_raw[event_target]

        X_test_raw = test_raw.drop(columns=[time_target, event_target])
        y_test_time = test_raw[time_target]
        y_test_event = test_raw[event_target]

        # Step 4.3 & 4.4 & 4.5: Fit Imputation, Encoding, and Scaling on Train ONLY
        X_train_proc, X_val_proc, X_test_proc = self._fit_transform_features(
            X_train_raw, X_val_raw, X_test_raw
        )

        # Assemble final dataframes
        train_processed = X_train_proc.copy()
        train_processed["time"] = y_train_time.values
        train_processed["event"] = y_train_event.values

        val_processed = X_val_proc.copy()
        val_processed["time"] = y_val_time.values
        val_processed["event"] = y_val_event.values

        test_processed = X_test_proc.copy()
        test_processed["time"] = y_test_time.values
        test_processed["event"] = y_test_event.values

        # Step 4.11: Generate 5-Fold Stratified Cross-Validation Folds on Train set
        cv_folds = self.splitter.create_cv_folds(train_processed, event_col="event", n_splits=5)

        # Step 4.12: Export Processed Datasets & Metadata
        save_dir = os.path.join(output_dir, "processed", self.dataset_name)
        os.makedirs(save_dir, exist_ok=True)

        train_path = os.path.join(save_dir, "train.csv")
        val_path = os.path.join(save_dir, "val.csv")
        test_path = os.path.join(save_dir, "test.csv")
        cv_path = os.path.join(save_dir, "cv_folds.json")
        meta_path = os.path.join(save_dir, "metadata.json")

        train_processed.to_csv(train_path, index=False)
        val_processed.to_csv(val_path, index=False)
        test_processed.to_csv(test_path, index=False)

        with open(cv_path, "w", encoding="utf-8") as f:
            json.dump(cv_folds, f, indent=2)

        metadata = {
            "dataset_name": self.dataset_name,
            "random_state": self.random_state,
            "target_time_col": "time",
            "target_event_col": "event",
            "num_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "train_samples": len(train_processed),
            "val_samples": len(val_processed),
            "test_samples": len(test_processed),
            "train_censoring_rate": float(round((1 - train_processed["event"].mean()) * 100, 2)),
            "val_censoring_rate": float(round((1 - val_processed["event"].mean()) * 100, 2)),
            "test_censoring_rate": float(round((1 - test_processed["event"].mean()) * 100, 2)),
            "scaling_params": self.scaling_params,
            "encoding_maps": self.encoding_maps,
            "validation_report": val_report
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return metadata

    def _fit_transform_features(
        self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fits imputation, encoding, and scaling strictly on Train, transforms Val and Test."""

        # 1. Missing Value Imputation
        for col in X_train.columns:
            if X_train[col].dtype in ['int64', 'float64']:
                med_val = X_train[col].median()
                self.imputation_values[col] = float(med_val)
                X_train[col] = X_train[col].fillna(med_val)
                X_val[col] = X_val[col].fillna(med_val)
                X_test[col] = X_test[col].fillna(med_val)
            else:
                mode_val = X_train[col].mode()[0] if not X_train[col].mode().empty else "missing"
                self.imputation_values[col] = str(mode_val)
                X_train[col] = X_train[col].fillna(mode_val)
                X_val[col] = X_val[col].fillna(mode_val)
                X_test[col] = X_test[col].fillna(mode_val)

        # 2. Categorical One-Hot Encoding
        cat_cols = [c for c in self.categorical_cols if c in X_train.columns]
        
        encoded_train_parts = []
        encoded_val_parts = []
        encoded_test_parts = []

        num_cols = [c for c in X_train.columns if c not in cat_cols and c not in self.ordinal_cols]

        # Pass numericals through
        encoded_train_parts.append(X_train[num_cols].copy())
        encoded_val_parts.append(X_val[num_cols].copy())
        encoded_test_parts.append(X_test[num_cols].copy())

        # Ordinal encoding if specified
        for ord_col, ord_order in self.ordinal_cols.items():
            if ord_col in X_train.columns:
                ord_map = {val: idx for idx, val in enumerate(ord_order)}
                self.encoding_maps[ord_col] = ord_map

                encoded_train_parts.append(pd.DataFrame({ord_col: X_train[ord_col].map(ord_map).fillna(0)}, index=X_train.index))
                encoded_val_parts.append(pd.DataFrame({ord_col: X_val[ord_col].map(ord_map).fillna(0)}, index=X_val.index))
                encoded_test_parts.append(pd.DataFrame({ord_col: X_test[ord_col].map(ord_map).fillna(0)}, index=X_test.index))

        # One-hot encoding for nominal categoricals
        for cat_col in cat_cols:
            # Fit categories on train
            unique_cats = sorted([str(x) for x in X_train[cat_col].unique()])
            # Drop first category to avoid multicollinearity
            dummy_cats = unique_cats[1:] if len(unique_cats) > 1 else unique_cats
            self.encoding_maps[cat_col] = dummy_cats

            train_dummies = pd.get_dummies(X_train[cat_col].astype(str), prefix=cat_col)
            val_dummies = pd.get_dummies(X_val[cat_col].astype(str), prefix=cat_col)
            test_dummies = pd.get_dummies(X_test[cat_col].astype(str), prefix=cat_col)

            for d_cat in dummy_cats:
                col_name = f"{cat_col}_{d_cat}"
                encoded_train_parts.append(pd.DataFrame({col_name: train_dummies.get(col_name, 0).astype(float)}, index=X_train.index))
                encoded_val_parts.append(pd.DataFrame({col_name: val_dummies.get(col_name, 0).astype(float)}, index=X_val.index))
                encoded_test_parts.append(pd.DataFrame({col_name: test_dummies.get(col_name, 0).astype(float)}, index=X_test.index))

        df_train_enc = pd.concat(encoded_train_parts, axis=1)
        df_val_enc = pd.concat(encoded_val_parts, axis=1)
        df_test_enc = pd.concat(encoded_test_parts, axis=1)

        # 3. Numerical Scaling (StandardScaler fitted strictly on Train set)
        continuous_cols = [c for c in df_train_enc.columns if df_train_enc[c].nunique() > 2]
        
        df_train_scaled = df_train_enc.copy()
        df_val_scaled = df_val_enc.copy()
        df_test_scaled = df_test_enc.copy()

        for c_col in continuous_cols:
            mean_v = float(df_train_enc[c_col].mean())
            std_v = float(df_train_enc[c_col].std(ddof=1))
            if std_v == 0 or np.isnan(std_v):
                std_v = 1.0

            self.scaling_params[c_col] = {"mean": mean_v, "std": std_v}

            df_train_scaled[c_col] = (df_train_enc[c_col] - mean_v) / std_v
            df_val_scaled[c_col] = (df_val_enc[c_col] - mean_v) / std_v
            df_test_scaled[c_col] = (df_test_enc[c_col] - mean_v) / std_v

        self.feature_names = list(df_train_scaled.columns)

        return df_train_scaled, df_val_scaled, df_test_scaled
