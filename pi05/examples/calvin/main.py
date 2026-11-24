"""Launch a WeaveVLA CALVIN evaluation."""
import os
from pathlib import Path
import sys

if __name__ == "__main__":
    launcher = Path(__file__).resolve().parents[2] / "scripts/evaluate.py"
    os.execv(sys.executable, [sys.executable, str(launcher), "calvin", *sys.argv[1:]])
