import json
from pathlib import Path


def main():
    cells_file = Path(
        "reports/confirmatory/confirmatory_v1_confirmatory_cdcdcee84651/frozen_cells.json"
    )
    if not cells_file.exists():
        print("Audit FAIL: frozen_cells.json not found")
        return

    with open(cells_file) as f:
        cells = json.load(f)

    expected_count = 6 * 3 * 25  # 450
    if len(cells) != expected_count:
        print(f"Audit FAIL: Expected {expected_count} cells, found {len(cells)}")
        return

    for cell in cells:
        if cell["status"] not in ["complete", "failed_diagnostics", "failed"]:
            print(
                f"Audit FAIL: Cell {cell['cell_id']} has invalid status {cell['status']}"
            )
            return

    print("Audit PASS: All 450 expected cells accounted for.")


if __name__ == "__main__":
    main()
