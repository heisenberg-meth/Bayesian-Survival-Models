"""
CLI entry point for running survival model experiments and pipelines.
"""
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Bayesian Cox Survival Analysis Command Line Interface")
    parser.add_argument("--config", type=str, help="Path to experiment configuration YAML file")
    parser.add_argument("--dataset", type=str, help="Dataset name (gbsg2, metabric, whas500)")
    parser.add_argument("--model", type=str, default="bayesian_cox", help="Model type to train")

    args = parser.parse_args()
    print(f"Running Bayesian Survival Model CLI with arguments: {args}")

if __name__ == "__main__":
    main()
