import csv
import os
from collections import Counter

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw"
)

datasets = {
    "GBSG2": ("gbsg2.csv", "time", "cens"),
    "WHAS500": ("whas500.csv", "lenfol", "fstat"),
    "METABRIC": ("metabric.csv", "duration", "event"),
}

for name, (fname, time_col, event_col) in datasets.items():
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        print(f"File {fname} not found!")
        continue

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    print(f"=== {name} ===")
    print(f"File path: {path}")
    print(f"Shape: {len(rows)} rows, {len(header)} columns")
    print(f"Columns: {header}")
    print(f"Target Time: {time_col}, Target Event: {event_col}")

    # Check target event values
    time_idx = header.index(time_col)
    event_idx = header.index(event_col)

    events = [int(r[event_idx]) for r in rows]
    times = [float(r[time_idx]) for r in rows]

    event_counts = Counter(events)
    print(f"Event counts (1=event, 0=censored): {dict(event_counts)}")
    censored_count = event_counts.get(0, 0)
    censoring_rate = censored_count / len(rows)
    print(f"Censoring rate: {censoring_rate:.4f} ({censoring_rate * 100:.2f}%)")
    print(
        f"Time min: {min(times)}, max: {max(times)}, avg: {sum(times) / len(times):.2f}\n"
    )
