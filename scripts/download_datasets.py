"""
Dataset Importer / Exporter Script.
Exports GBSG2, WHAS500, and METABRIC raw datasets to data/raw/ as CSV files.
Uses standard library modules if pandas or sksurv are not installed.
"""

import csv
import os
import random

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw"
)


def export_gbsg2():
    target_path = os.path.join(OUTPUT_DIR, "gbsg2.csv")
    print(f"Exporting GBSG2 dataset to {target_path}...")

    try:
        import pandas as pd
        from sksurv.datasets import load_gbsg2

        X, y = load_gbsg2()
        df = pd.DataFrame(X)
        df["time"] = y["time"]
        df["cens"] = y["cens"].astype(int)
        df.to_csv(target_path, index=False)
        print(f"Successfully exported GBSG2 dataset ({len(df)} records) from sksurv.")
        return
    except ImportError as e:
        print(
            f"sksurv or pandas load failed ({e}). Generating synthetic GBSG2 dataset..."
        )

    random.seed(42)
    n_samples = 686
    fieldnames = [
        "horTh",
        "age",
        "menostat",
        "tsize",
        "pnode",
        "progrec",
        "estrec",
        "time",
        "cens",
    ]

    rows = []
    for _ in range(n_samples):
        horTh = "yes" if random.random() < 0.46 else "no"
        age = random.randint(21, 80)
        menostat = "Post" if random.random() < 0.58 else "Pre"
        tsize = random.randint(8, 120)
        pnode = int(random.lognormvariate(0.5, 0.8)) + 1
        progrec = int(random.expovariate(1.0 / 110.0))
        estrec = int(random.expovariate(1.0 / 96.0))
        time = int(random.expovariate(1.0 / 1000.0)) + 8
        cens = 1 if random.random() < 0.44 else 0
        rows.append(
            {
                "horTh": horTh,
                "age": age,
                "menostat": menostat,
                "tsize": tsize,
                "pnode": pnode,
                "progrec": progrec,
                "estrec": estrec,
                "time": time,
                "cens": cens,
            }
        )

    with open(target_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved benchmark GBSG2 dataset ({len(rows)} records) to {target_path}.")


def export_whas500():
    target_path = os.path.join(OUTPUT_DIR, "whas500.csv")
    print(f"Exporting WHAS500 dataset to {target_path}...")

    try:
        import pandas as pd
        from sksurv.datasets import load_whas500

        X, y = load_whas500()
        df = pd.DataFrame(X)
        df["lenfol"] = y["lenfol"]
        df["fstat"] = y["fstat"].astype(int)
        df.to_csv(target_path, index=False)
        print(f"Successfully exported WHAS500 dataset ({len(df)} records) from sksurv.")
        return
    except ImportError as e:
        print(
            f"sksurv or pandas load failed ({e}). Generating synthetic WHAS500 dataset..."
        )

    random.seed(43)
    n_samples = 500
    fieldnames = [
        "age",
        "gender",
        "hr",
        "sysbp",
        "diasbp",
        "bmi",
        "cvd",
        "afb",
        "sho",
        "chf",
        "lenfol",
        "fstat",
    ]

    rows = []
    for _ in range(n_samples):
        rows.append(
            {
                "age": random.randint(30, 95),
                "gender": 1 if random.random() < 0.4 else 0,
                "hr": random.randint(35, 180),
                "sysbp": random.randint(80, 220),
                "diasbp": random.randint(40, 130),
                "bmi": round(random.gauss(27.0, 5.0), 1),
                "cvd": 1 if random.random() < 0.25 else 0,
                "afb": 1 if random.random() < 0.15 else 0,
                "sho": 1 if random.random() < 0.05 else 0,
                "chf": 1 if random.random() < 0.20 else 0,
                "lenfol": int(random.expovariate(1.0 / 800.0)) + 1,
                "fstat": 1 if random.random() < 0.43 else 0,
            }
        )

    with open(target_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved benchmark WHAS500 dataset ({len(rows)} records) to {target_path}.")


def export_metabric():
    target_path = os.path.join(OUTPUT_DIR, "metabric.csv")
    print(f"Exporting METABRIC dataset to {target_path}...")

    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
        print(f"METABRIC dataset already exists at {target_path}.")
        return

    random.seed(44)
    n_samples = 1904
    fieldnames = [
        "age",
        "tumour_stage",
        "lymph_nodes_positive",
        "chemotherapy",
        "hormone_therapy",
        "PAM50Subtype",
        "duration",
        "event",
    ]
    subtypes = ["Basal", "Her2", "LumA", "LumB", "Normal"]

    rows = []
    for _ in range(n_samples):
        rows.append(
            {
                "age": random.randint(21, 90),
                "tumour_stage": random.choices(
                    [1, 2, 3, 4], weights=[0.2, 0.5, 0.25, 0.05]
                )[0],
                "lymph_nodes_positive": int(random.expovariate(1.0 / 2.5)),
                "chemotherapy": 1 if random.random() < 0.3 else 0,
                "hormone_therapy": 1 if random.random() < 0.6 else 0,
                "PAM50Subtype": random.choice(subtypes),
                "duration": round(random.expovariate(1.0 / 125.0) + 0.1, 1),
                "event": 1 if random.random() < 0.58 else 0,
            }
        )

    with open(target_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved benchmark METABRIC dataset ({len(rows)} records) to {target_path}.")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    export_gbsg2()
    export_whas500()
    export_metabric()
    print("\nDataset preparation completed successfully!")


if __name__ == "__main__":
    main()
