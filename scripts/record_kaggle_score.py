"""Record Kaggle's returned score, with a downloaded-file hash and screenshot.

Requires an authenticated current Kaggle CLI. Uses documented commands from
https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md .
This script does not submit anything and never accepts a hand-entered score.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exam.data import json_write, sha256
from exam.validation import verify_results

COMPETITION = "sweettv-tv-program-recommender"


def parse_returned_score(raw: str, reference: int) -> dict:
    rows = list(csv.DictReader(io.StringIO(raw)))
    for original in rows:
        row = {
            key.lower().replace("_", "").replace(" ", ""): value
            for key, value in original.items()
            if key
        }
        if row.get("ref", row.get("id")) != str(reference):
            continue
        if "complete" not in row.get("status", "").lower():
            raise ValueError("Kaggle submission is not COMPLETE")
        private = row.get("privatescore", "")
        if private in {"", "None", "null"}:
            raise ValueError("Kaggle has not returned a private score")
        private = float(private)
        public = row.get("publicscore", "")
        if not 0 <= private <= 1:
            raise ValueError("Invalid returned MAP score")
        return {
            "submission_ref": reference,
            "private_score": private,
            "public_score": float(public) if public not in {"", "None", "null"} else None,
            "returned_row": original,
        }
    raise ValueError(
        "Designated submission not found in returned records; inspect Kaggle submission paging"
    )


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--submission-ref", type=int, required=True)
    p.add_argument(
        "--screenshot",
        type=Path,
        required=True,
        help="Actual Kaggle UI screenshot showing private score and designated ref",
    )
    p.add_argument("--submission", type=Path, default=Path("exam/submissions/final.csv"))
    p.add_argument("--cache-dir", type=Path, default=Path("exam/cache"))
    p.add_argument("--results-dir", type=Path, default=Path("exam/results"))
    p.add_argument("--output", type=Path, default=Path("exam/results/kaggle_evidence.json"))
    args = p.parse_args(argv)
    if not shutil.which("kaggle"):
        raise RuntimeError("Install/authenticate the Kaggle CLI in this local environment")
    if not args.screenshot.is_file():
        raise FileNotFoundError("Actual Kaggle screenshot is required")
    verify_results(args.cache_dir, args.results_dir, args.submission)
    summary = json.loads((args.results_dir / "summary.json").read_text())
    if summary["provenance"]["data_kind"] != "competition":
        raise ValueError("Synthetic results cannot be submitted as exam evidence")
    response = subprocess.run(
        ["kaggle", "competitions", "submissions", COMPETITION, "--csv"],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    score = parse_returned_score(response.stdout, args.submission_ref)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["kaggle", "competitions", "submission-download", str(args.submission_ref), "-p", tmp],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        csv_files = list(Path(tmp).rglob("*.csv"))
        matching = [path for path in csv_files if sha256(path) == sha256(args.submission)]
        if len(matching) != 1:
            raise ValueError(
                "Downloaded Kaggle submission is not byte-identical to the local final.csv"
            )
    saved_image = args.output.parent / ("kaggle_score" + args.screenshot.suffix.lower())
    saved_image.parent.mkdir(parents=True, exist_ok=True)
    if saved_image.resolve() != args.screenshot.resolve():
        shutil.copyfile(args.screenshot, saved_image)
    score.update(
        {
            "status": "COMPLETE",
            "competition": COMPETITION,
            "recorded_utc": datetime.now(timezone.utc).isoformat(),
            "submission_sha256": sha256(args.submission),
            "downloaded_file_matches": True,
            "screenshot": saved_image.name,
            "screenshot_sha256": sha256(saved_image),
            "provenance": summary["provenance"],
            "source": "authenticated Kaggle CLI submissions response",
        }
    )
    json_write(args.output, score)
    (args.output.parent / "kaggle_returned_submissions.csv").write_text(response.stdout)
    print(json.dumps(score, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
