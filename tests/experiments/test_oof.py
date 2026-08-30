import json

from src.experiments.oof import aggregate_oof_predictions


def test_aggregate_oof_predictions(tmp_path):
    d1 = tmp_path / "cell_1"
    d1.mkdir()
    with (d1 / "result.json").open("w") as f:
        json.dump(
            {
                "fold": 0,
                "subject_ids": ["S1", "S2"],
                "time": [10, 20],
                "event": [1, 0],
                "predictions": [1.5, 0.5],
                "survival": [[0.9, 0.8], [0.95, 0.9]],
            },
            f,
        )

    d2 = tmp_path / "cell_2"
    d2.mkdir()
    with (d2 / "result.json").open("w") as f:
        json.dump(
            {
                "fold": 1,
                "subject_ids": ["S3", "S4"],
                "time": [15, 25],
                "event": [0, 1],
                "predictions": [0.8, 1.2],
                "survival": [[0.92, 0.85], [0.88, 0.75]],
            },
            f,
        )

    df = aggregate_oof_predictions(str(tmp_path))
    assert len(df) == 4
    assert set(df["subject_id"]) == {"S1", "S2", "S3", "S4"}

    # Check duplicate OOF predictions
    assert df["subject_id"].duplicated().sum() == 0
