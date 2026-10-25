"""Run from the repository root, in local Codex or an ordinary local terminal."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exam.data import preflight, prepare
from exam.experiments import predict, run_search
from exam.validation import verify_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("exam/data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("exam/cache"))
    parser.add_argument("--results-dir", type=Path, default=Path("exam/results"))
    parser.add_argument("--submission", type=Path, default=Path("exam/submissions/final.csv"))
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--overlap-policy", choices=["union", "sum"], default="union")
    args = parser.parse_args()
    status = preflight(args.data_dir)
    if not status["ready"]:
        print(json.dumps(status, indent=2))
        return 2
    if args.rebuild_cache or not (args.cache_dir / "manifest.json").is_file():
        prepare(args.data_dir, args.cache_dir, overlap_policy=args.overlap_policy)
    else:
        manifest = json.loads((args.cache_dir / "manifest.json").read_text())
        if manifest["configuration"]["overlap_policy"] != args.overlap_policy:
            raise ValueError(
                "Cache overlap policy differs; rebuild and use a NEW results directory"
            )
    run_search(
        args.cache_dir, args.results_dir, resume=(args.results_dir / "trial_plan.json").is_file()
    )
    predict(args.cache_dir, args.results_dir, args.submission)
    verified = verify_results(args.cache_dir, args.results_dir, args.submission)
    print(json.dumps(verified, indent=2))
    print(
        "Local experiments and CSV are complete. Kaggle submission and the native session export remain separate steps."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
