"""Install the bundled OpenPI Torch adapter files into this Python environment."""
import argparse
from importlib.metadata import distribution
from pathlib import Path
from tempfile import TemporaryDirectory
from shutil import copy2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    transformers = distribution("transformers")
    if transformers.version != "4.53.2":
        parser.error(f"Expected Transformers 4.53.2, found {transformers.version}")
    source = Path(__file__).resolve().parents[1] / "src/openpi/models_pytorch/transformers_replace"
    installed = transformers.locate_file("transformers")
    if not (installed / "models").is_dir():
        parser.error(f"Transformers model directory missing: {installed}")
    for patch in sorted(source.rglob("*.py")):
        destination = installed / patch.relative_to(source)
        with TemporaryDirectory(dir=destination.parent, prefix=".openpi-patch-") as staging:
            staged = Path(staging) / destination.name
            copy2(patch, staged)
            staged.replace(destination)
        print(destination.relative_to(installed))


if __name__ == "__main__":
    main()
