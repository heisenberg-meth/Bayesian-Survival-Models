"""
Feature Engineering Module for Bayesian Survival Models.
Implements clinically and statistically justified transformations and interaction terms.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

class FeatureEngineer:
    """Applies clinical feature engineering transformations to survival datasets."""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name.lower()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies feature transformations to dataframe."""
        df_out = df.copy()

        if self.dataset_name == "gbsg2":
            # 1. Log transform right-skewed counts/receptor concentrations
            if "pnode" in df_out.columns:
                df_out["log_pnode"] = np.log1p(df_out["pnode"])
            if "progrec" in df_out.columns:
                df_out["log_progrec"] = np.log1p(df_out["progrec"])
            if "estrec" in df_out.columns:
                df_out["log_estrec"] = np.log1p(df_out["estrec"])

            # 2. Clinical interaction: Age x Hormone Therapy
            if "age" in df_out.columns and "horTh" in df_out.columns:
                hor_val = df_out["horTh"].apply(lambda x: 1 if str(x).lower() in ("yes", "1") else 0)
                df_out["age_x_horTh"] = df_out["age"] * hor_val

        elif self.dataset_name == "whas500":
            # 1. Log transform follow-up length (optional/clinical)
            # 2. Clinical interaction: Age x Congestive Heart Failure
            if "age" in df_out.columns and "chf" in df_out.columns:
                df_out["age_x_chf"] = df_out["age"] * df_out["chf"]
            if "age" in df_out.columns and "gender" in df_out.columns:
                df_out["age_x_gender"] = df_out["age"] * df_out["gender"]

        elif self.dataset_name == "metabric":
            # 1. Log transform positive lymph nodes
            if "lymph_nodes_positive" in df_out.columns:
                df_out["log_lymph_nodes"] = np.log1p(df_out["lymph_nodes_positive"])

            # 2. Clinical interaction: Age x Tumour Stage
            if "age" in df_out.columns and "tumour_stage" in df_out.columns:
                df_out["age_x_stage"] = df_out["age"] * df_out["tumour_stage"]

        return df_out
