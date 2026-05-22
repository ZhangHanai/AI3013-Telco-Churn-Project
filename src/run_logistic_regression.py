from pathlib import Path
import argparse

from src.logistic_regression_scratch import run_experiment


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["demo", "full"], default="full")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(Path.cwd(), mode=args.mode)
