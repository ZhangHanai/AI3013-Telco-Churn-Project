from pathlib import Path
from src.logistic_regression_scratch import run_experiment

if __name__ == "__main__":
    run_experiment(Path(__file__).resolve().parent)
