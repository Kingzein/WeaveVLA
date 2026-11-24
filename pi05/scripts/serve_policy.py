"""Run a local WeaveVLA policy worker for a benchmark client."""
import argparse
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    benchmarks = parser.add_subparsers(dest="benchmark", required=True)
    benchmarks.add_parser("libero", help="Serve the local socket configured by the LIBERO launcher")
    calvin = benchmarks.add_parser("calvin", help="Serve an inherited pipe from the CALVIN launcher")
    calvin.add_argument("--pipe-fd", type=int, required=True)
    calvin.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    if args.benchmark == "libero":
        runpy.run_module("weavevla.serving.libero.server", run_name="__main__")
    else:
        from weavevla.serving.calvin.policy_service import serve
        serve(args.pipe_fd, args.checkpoint)


if __name__ == "__main__":
    main()
