from pathlib import Path
import subprocess
import sys


def main():
    project_root = Path(__file__).resolve().parent
    result = subprocess.run([sys.executable, "-m", "src.run_knn_experiment", "--fast"], cwd=project_root, text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
