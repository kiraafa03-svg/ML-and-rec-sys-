"""Run software checks and write an honestly scoped, machine-readable record."""

import argparse
import io
import json
import platform
import sys
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exam.data import json_write
from exam.experiments import source_fingerprint

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=Path("exam/results/software_checks.json"))
args = parser.parse_args()
t0 = time.perf_counter()
stream = io.StringIO()
suite = unittest.defaultTestLoader.discover("exam/tests")
result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
record = {
    "scope": "SOFTWARE TESTS ONLY. Includes 150 configurations on a small synthetic fixture; no real Kaggle experiment or score.",
    "data_kind": "synthetic_fixture",
    "passed": result.wasSuccessful(),
    "test_cases": result.testsRun,
    "failures": len(result.failures),
    "errors": len(result.errors),
    "seconds": time.perf_counter() - t0,
    "python": platform.python_version(),
    "platform": platform.platform(),
    "source_fingerprint": source_fingerprint(),
    "recorded_utc": datetime.now(timezone.utc).isoformat(),
    "output": stream.getvalue(),
}
json_write(args.output, record)
print(json.dumps({k: v for k, v in record.items() if k != "output"}, indent=2))
if not result.wasSuccessful():
    print(stream.getvalue())
raise SystemExit(0 if result.wasSuccessful() else 1)
