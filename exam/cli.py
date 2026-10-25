"""Command line entry point. No credentials are printed or committed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .data import preflight, prepare
from .experiments import predict, run_search, trial_plan
from .validation import verify_results


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="sweet.tv exam: reproducible temporal recommendation")
    sub = p.add_subparsers(dest="command", required=True)
    check = sub.add_parser("preflight", help="Check that all five competition files exist")
    check.add_argument("--data-dir", type=Path, default=Path("exam/data"))
    prep = sub.add_parser("prepare", help="Parse, clip and join raw views/schedules")
    prep.add_argument("--data-dir", type=Path, default=Path("exam/data"))
    prep.add_argument("--cache-dir", type=Path, default=Path("exam/cache"))
    prep.add_argument("--min-fraction", type=float, default=0.8)
    prep.add_argument("--overlap-policy", choices=["union", "sum"], default="union")
    prep.add_argument(
        "--fixture",
        action="store_true",
        help="Explicitly label synthetic test inputs; never an exam result",
    )
    tune = sub.add_parser("tune", help="Run/resume the predeclared 150-configuration search")
    tune.add_argument("--cache-dir", type=Path, default=Path("exam/cache"))
    tune.add_argument("--output-dir", type=Path, default=Path("exam/results"))
    tune.add_argument("--resume", action="store_true")
    tune.add_argument("--min-truth-items", type=int, default=5)
    tune.add_argument(
        "--limit",
        type=int,
        default=150,
        help="Smoke-test prefix only; final prediction still requires all 150",
    )
    tune.add_argument("--allow-fixture", action="store_true")
    final = sub.add_parser("predict", help="Refit selected strategy and write the final CSV")
    final.add_argument("--cache-dir", type=Path, default=Path("exam/cache"))
    final.add_argument("--results-dir", type=Path, default=Path("exam/results"))
    final.add_argument("--submission", type=Path, default=Path("exam/submissions/final.csv"))
    final.add_argument("--allow-fixture", action="store_true")
    verify = sub.add_parser(
        "verify", help="Independently check all experiment and submission artifacts"
    )
    verify.add_argument("--cache-dir", type=Path, default=Path("exam/cache"))
    verify.add_argument("--results-dir", type=Path, default=Path("exam/results"))
    verify.add_argument("--submission", type=Path)
    sub.add_parser("plan", help="Print the exact 150 configurations without running them")
    report = sub.add_parser(
        "report", help="Generate the exam PDF from verified real results and Kaggle evidence"
    )
    report.add_argument("--cache-dir", type=Path, default=Path("exam/cache"))
    report.add_argument("--results-dir", type=Path, default=Path("exam/results"))
    report.add_argument("--submission", type=Path, default=Path("exam/submissions/final.csv"))
    report.add_argument(
        "--evidence",
        type=Path,
        required=True,
        help="Kaggle evidence JSON from record_kaggle_score.py",
    )
    report.add_argument(
        "--output", type=Path, default=Path("exam/report/Kirill_Afanasyev_exam_report.pdf")
    )
    report.add_argument("--student", default="Kirill Afanasyev")
    report.add_argument("--email", default="", help="Use your own course email")
    report.add_argument(
        "--personal-verification", type=Path, help="Your own written verification statement"
    )
    report.add_argument(
        "--session-file", type=Path, required=True, help="Original coding-agent session export"
    )
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "preflight":
            result = preflight(args.data_dir)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["ready"] else 2
        if args.command == "prepare":
            result = prepare(
                args.data_dir, args.cache_dir, args.min_fraction, args.overlap_policy, args.fixture
            )
        elif args.command == "tune":
            result = run_search(
                args.cache_dir,
                args.output_dir,
                args.resume,
                args.min_truth_items,
                args.limit,
                args.allow_fixture,
            )
        elif args.command == "predict":
            result = predict(args.cache_dir, args.results_dir, args.submission, args.allow_fixture)
        elif args.command == "verify":
            result = verify_results(args.cache_dir, args.results_dir, args.submission)
        elif args.command == "plan":
            result = {"count": 150, "trials": trial_plan()}
        elif args.command == "report":
            from .report import create_report

            result = create_report(args)
        else:
            raise AssertionError("Unreachable command")
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
