from pathlib import Path
import subprocess
import sys


def run(cmd, cwd):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main():
    project_root = Path(__file__).resolve().parent
    run([sys.executable, "-m", "src.run_knn_experiment"], project_root)
    run([sys.executable, "-m", "src.run_svm_experiments"], project_root)


if __name__ == "__main__":
    main()
