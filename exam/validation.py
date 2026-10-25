"""Independent artifact checks, including strict submission identity/order."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .data import ground_truth, json_write, load_dataset, sha256
from .experiments import config_hash, source_fingerprint, trial_plan
from .metrics import apk


def verify_submission(path: Path, template_users: np.ndarray, pool: np.ndarray) -> dict:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["user_id", "tv_show_id"]:
            raise ValueError("Submission column names/order differ")
        rows = list(reader)
    if len(rows) != len(template_users):
        raise ValueError("Submission row count differs from template")
    if [r["user_id"] for r in rows] != [str(int(user)) for user in template_users]:
        raise ValueError("User IDs or row order differ from the original template")
    candidates = set(map(int, pool))
    for number, row in enumerate(rows, 2):
        tokens = row["tv_show_id"].split()
        try:
            items = [int(token) for token in tokens]
        except ValueError as exc:
            raise ValueError(f"Row {number}: noninteger item ID") from exc
        if len(items) != 5 or len(set(items)) != 5 or 0 in items or not set(items) <= candidates:
            raise ValueError(
                f"Row {number}: expected five distinct nonzero scheduled programme IDs"
            )
    return {
        "passed": True,
        "rows": len(rows),
        "sha256": sha256(path),
        "candidate_programmes": len(candidates),
    }


def independent_ap(actual, predicted) -> float:
    """A second, vector-free formulation used only as an oracle."""
    actual = set(actual)
    if not actual:
        return 0.0
    hit_positions = []
    used = set()
    for rank, show in enumerate(predicted[:5], 1):
        if show in actual and show not in used:
            hit_positions.append(rank)
        used.add(show)
    return sum(number / rank for number, rank in enumerate(hit_positions, 1)) / min(5, len(actual))


def verify_results(cache_dir: Path, results_dir: Path, submission: Path | None = None) -> dict:
    summary = json.loads((results_dir / "summary.json").read_text())
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    if summary["provenance"]["source_fingerprint"] != source_fingerprint():
        raise ValueError("Source changed since search")
    if summary["provenance"]["data_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Cache/experiment data mismatch")
    plan = trial_plan()
    records = []
    for config in plan:
        path = results_dir / "trials" / (config["trial_id"] + ".json")
        if not path.is_file():
            raise ValueError(f"Missing experiment: {config['trial_id']}")
        record = json.loads(path.read_text())
        if (
            record["status"] != "complete"
            or record["config"] != config
            or record["config_hash"] != config_hash(config)
        ):
            raise ValueError(f"Incomplete/modified experiment: {config['trial_id']}")
        if record["provenance"] != summary["provenance"]:
            raise ValueError("Mixed experiment provenance")
        if not 0 <= record["map5"] <= 1:
            raise ValueError("MAP outside [0,1]")
        records.append(record)
    truth, cohort = ground_truth(
        np.load(cache_dir / "validation_counts.npy", allow_pickle=False),
        summary["provenance"]["min_truth_items"],
    )
    maps = {}
    for method in ["popularity", "ppr", "mf", "selected"]:
        lists = json.loads((results_dir / "predictions" / (method + ".json")).read_text())
        values = []
        for user, actual in truth.items():
            predicted = lists[str(user)]
            a, b = apk(actual, predicted), independent_ap(actual, predicted)
            if abs(a - b) > 1e-12:
                raise AssertionError("AP implementations disagree")
            values.append(b)
        maps[method] = float(np.mean(values))
    if abs(maps["selected"] - summary["selected_map5"]) > 1e-10:
        raise ValueError("Selected predictions disagree with summary")
    report = {
        "passed": True,
        "scope": "artifact integrity and independent AP; not a claim of personal student verification",
        "data_kind": manifest["data_kind"],
        "experiments": 150,
        "cohort": cohort,
        "independent_map5": maps,
        "provenance": summary["provenance"],
    }
    if submission is not None:
        data = load_dataset(cache_dir, final=True)
        checked = verify_submission(
            submission, np.load(cache_dir / "template_users.npy", allow_pickle=False), data.pool
        )
        final = json.loads((results_dir / "final_prediction.json").read_text())
        if final["sha256"] != checked["sha256"]:
            raise ValueError("Submission changed after prediction")
        report["submission"] = checked
    json_write(results_dir / "verification.json", report)
    return report
