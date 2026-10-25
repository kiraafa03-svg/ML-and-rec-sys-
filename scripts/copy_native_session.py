"""Copy an existing original export byte-for-byte; never reconstruct it."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exam.session import copy_native_session

parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
parser.add_argument("destination", type=Path)
args = parser.parse_args()
try:
    print(json.dumps(copy_native_session(args.source, args.destination), indent=2))
except (OSError, ValueError) as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    raise SystemExit(2)
