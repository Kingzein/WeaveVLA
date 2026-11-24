"""Train a WeaveVLA configuration from the π0.5 base checkpoint."""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from weavevla.training.train import main

if __name__ == "__main__":
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    main()
