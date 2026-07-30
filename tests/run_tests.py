import importlib
import os
import sys
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    print("=" * 60)
    print("RUNNING CUSTOM TEST DISCOVERY & EXECUTION")
    print("=" * 60)

    tests_dir = os.path.join(PROJECT_ROOT, "tests")
    test_files = [
        f[:-3]
        for f in os.listdir(tests_dir)
        if f.startswith("test_") and f.endswith(".py")
    ]

    total_run = 0
    total_failed = 0

    for module_name in sorted(test_files):
        print(f"\n[+] Scanning module: tests.{module_name}")
        try:
            mod = importlib.import_module(f"tests.{module_name}")
        except Exception as e:
            print(f"  [!] Failed to import tests.{module_name}: {e}")
            traceback.print_exc()
            total_failed += 1
            continue

        # Get all functions starting with test_
        test_funcs = [
            (name, getattr(mod, name))
            for name in dir(mod)
            if name.startswith("test_") and callable(getattr(mod, name))
        ]

        if not test_funcs:
            print("  No test functions found.")
            continue

        for name, func in test_funcs:
            print(f"  Running {name}... ", end="", flush=True)
            total_run += 1
            try:
                func()
                print("PASSED")
            except Exception as e:
                print("FAILED: ", e)
                traceback.print_exc()
                total_failed += 1

    print("\n" + "=" * 60)
    print(f"Test Summary: {total_run} run, {total_failed} failed.")
    print("=" * 60)
    if total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
