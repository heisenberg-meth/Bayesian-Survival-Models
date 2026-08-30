import json
from pathlib import Path

import pandas as pd


def aggregate_oof_predictions(checkpoint_dir: str) -> pd.DataFrame:
    """
    Aggregate Out-of-Fold predictions from an experiment checkpoint directory.
    Returns a DataFrame containing subject_id, fold, time, event, risk, and survival.
    """
    ckpt_dir = Path(checkpoint_dir)
    records = []

    for result_file in ckpt_dir.glob("**/result.json"):
        with result_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

            fold = data["fold"]
            subject_ids = data["subject_ids"]
            times = data["time"]
            events = data["event"]
            risks = data["predictions"]
            survivals = data["survival"]

            for i, subj in enumerate(subject_ids):
                records.append(
                    {
                        "subject_id": subj,
                        "fold": fold,
                        "time": times[i],
                        "event": events[i],
                        "risk": risks[i],
                        "survival": survivals[i],
                    }
                )

    return pd.DataFrame(records)
