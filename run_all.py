from pathlib import Path
import subprocess
import sys


def main():
    project_root = Path(__file__).resolve().parent
    script_path = project_root / "src" / "run_knn_experiment.py"
    result = subprocess.run([sys.executable, str(script_path)], cwd=project_root, text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
