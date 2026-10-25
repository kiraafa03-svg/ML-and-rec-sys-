"""Exactly 150 predeclared configurations, resumable with provenance checks."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .data import Dataset, ground_truth, json_write, load_dataset, sha256
from .metrics import apk, mapk
from .models import (
    WeightedALS,
    fallback_list,
    fuse,
    ppr_graph,
    ppr_scores,
    rank_top5,
    schedule_ratio,
    user_lookup,
)


def trial_plan() -> list[dict]:
    plan = []
    for alpha, transform, decay, topk in itertools.product(
        [0.1, 0.25, 0.4, 0.6, 0.85], ["raw", "log"], [0.0, 0.05], [100, 300]
    ):
        plan.append(
            {
                "method": "ppr",
                "alpha": alpha,
                "transform": transform,
                "decay": decay,
                "graph_topk": topk,
            }
        )
    for factors, confidence, regularization in itertools.product(
        [8, 16, 32, 48, 64], [1.0, 3.0, 10.0, 30.0, 100.0], [0.01, 0.05]
    ):
        plan.append(
            {
                "method": "mf",
                "factors": factors,
                "confidence": confidence,
                "regularization": regularization,
                "iterations": 20,
                "target": "binary",
                "seed": 42,
                "threads": 1,
            }
        )
    for factors, confidence in itertools.product([16, 32, 48, 64], [1.0, 3.0, 10.0, 30.0, 100.0]):
        plan.append(
            {
                "method": "mf",
                "factors": factors,
                "confidence": confidence,
                "regularization": 0.05,
                "iterations": 20,
                "target": "log",
                "seed": 42,
                "threads": 1,
            }
        )
    for weight, beta, kind in itertools.product(
        [0.25, 0.5, 0.75, 0.9, 0.98], [0.0, 0.15, 0.3, 0.6], ["minmax", "rrf"]
    ):
        plan.append({"method": "hybrid", "weight_mf": weight, "schedule_beta": beta, "kind": kind})
    assert len(plan) == 150
    assert len({json.dumps(c, sort_keys=True) for c in plan}) == 150
    return [{"trial_id": f"trial_{number:03d}", **config} for number, config in enumerate(plan, 1)]


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def source_fingerprint() -> str:
    h = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


def model_scores(data: Dataset, config: dict) -> tuple[np.ndarray, dict]:
    cfg = {k: v for k, v in config.items() if k not in {"method", "trial_id"}}
    if config["method"] == "ppr":
        matrix = data.weighted_matrix(cfg["decay"])
        graph, restart = ppr_graph(matrix, transform=cfg["transform"], topk=cfg["graph_topk"])
        scores, info = ppr_scores(graph, restart, alpha=cfg["alpha"])
        return scores, {**info, "graph_edges": int(graph.nnz)}
    if config["method"] == "mf":
        model = WeightedALS(**cfg).fit(data.matrix)
        return model.scores(), model.info
    raise ValueError("Component scores require ppr or mf")


def get_predictions(data: Dataset, scores: np.ndarray, users: np.ndarray, method: str):
    return rank_top5(scores, data, users, min_score=0.0 if method in {"ppr", "hybrid"} else -np.inf)


def save_predictions(path: Path, predictions: dict[int, list[int]]) -> None:
    json_write(path, {str(user): [int(x) for x in items] for user, items in predictions.items()})


def read_trial_records(output_dir: Path, provenance: dict) -> list[dict]:
    records = []
    for path in sorted((output_dir / "trials").glob("trial_*.json")):
        record = json.loads(path.read_text())
        if record.get("provenance") != provenance:
            raise ValueError(f"Stale trial provenance in {path}; choose a new output directory")
        records.append(record)
    return records


def run_search(
    cache_dir: Path,
    output_dir: Path,
    resume: bool = False,
    min_truth_items: int = 5,
    limit: int = 150,
    allow_fixture: bool = False,
) -> dict:
    if not 1 <= limit <= 150:
        raise ValueError("Trial limit must be 1..150")
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    if manifest["data_kind"] != "competition" and not allow_fixture:
        raise ValueError("Synthetic data cannot be used as a competition experiment")
    data = load_dataset(cache_dir)
    if data.matrix.nnz == 0:
        raise ValueError("No qualifying training interactions")
    truth, cohort = ground_truth(
        np.load(cache_dir / "validation_counts.npy", allow_pickle=False), min_truth_items
    )
    if not truth:
        raise ValueError("No evaluable validation users")
    users = np.array(list(truth), dtype=np.uint64)
    _, known = user_lookup(data.users, users)
    cohort["cold_start_users"] = int((~known).sum())
    provenance = {
        "data_fingerprint": manifest["fingerprint"],
        "source_fingerprint": source_fingerprint(),
        "min_truth_items": min_truth_items,
        "data_kind": manifest["data_kind"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = trial_plan()
    plan_doc = {
        "provenance": provenance,
        "trials": plan,
        "selection_metric": "MAP@5",
        "tie_break": "lower trial number",
        "scope": "validation selection score; not an unbiased test estimate",
    }
    plan_path = output_dir / "trial_plan.json"
    if plan_path.is_file():
        if not resume:
            raise FileExistsError("Results already exist; use --resume or a new output directory")
        if json.loads(plan_path.read_text()) != plan_doc:
            raise ValueError("Plan or provenance changed; use a fresh output directory")
    else:
        json_write(plan_path, plan_doc)
    previous = read_trial_records(output_dir, provenance)
    by_id = {r["config"]["trial_id"]: r for r in previous}
    for record in previous:
        cfg = next((c for c in plan if c["trial_id"] == record["config"]["trial_id"]), None)
        if cfg != record["config"] or record.get("config_hash") != config_hash(cfg):
            raise ValueError("Saved trial configuration was modified")
    fallback = fallback_list(data)
    baseline = {int(user): fallback[:5] for user in users}
    baseline_score = mapk(truth, baseline)
    save_predictions(output_dir / "predictions" / "popularity.json", baseline)
    json_write(
        output_dir / "baseline.json",
        {
            "method": "popularity",
            "map5": baseline_score,
            "cohort": cohort,
            "provenance": provenance,
        },
    )
    best_components = {}
    for method in ["ppr", "mf"]:
        candidates = [
            r for r in previous if r["config"]["method"] == method and r["status"] == "complete"
        ]
        if candidates:
            winner = max(
                candidates, key=lambda r: (r["map5"], -int(r["config"]["trial_id"].split("_")[-1]))
            )
            best_components[method] = winner
    graph_cache = {}
    component_scores = {}
    exposure = None
    started = time.perf_counter()
    for config in plan[:limit]:
        trial_id = config["trial_id"]
        if trial_id in by_id and by_id[trial_id]["status"] == "complete":
            continue
        t0 = time.perf_counter()
        method = config["method"]
        info = {}
        try:
            if method == "ppr":
                key = (config["transform"], config["decay"], config["graph_topk"])
                if key not in graph_cache:
                    graph_cache[key] = ppr_graph(
                        data.weighted_matrix(config["decay"]),
                        config["transform"],
                        config["graph_topk"],
                    )
                graph, restart = graph_cache[key]
                scores, info = ppr_scores(graph, restart, alpha=config["alpha"])
                if not info["converged"]:
                    raise RuntimeError("PPR did not converge within its cap")
                info["graph_edges"] = int(graph.nnz)
            elif method == "mf":
                # Graphs are no longer needed once the PPR stage is complete.
                graph_cache.clear()
                scores, info = model_scores(data, config)
            else:
                if set(best_components) != {"ppr", "mf"}:
                    raise RuntimeError("Hybrid requires both individually evaluated components")
                for component in ["ppr", "mf"]:
                    if component not in component_scores:
                        component_scores[component], _ = model_scores(
                            data, best_components[component]["config"]
                        )
                scores = fuse(
                    component_scores["ppr"],
                    component_scores["mf"],
                    data.items,
                    data.pool,
                    config["weight_mf"],
                    config["kind"],
                )
                if config["schedule_beta"]:
                    if exposure is None:
                        exposure = schedule_ratio(data)
                    scores *= exposure ** config["schedule_beta"]
                info = {
                    "ppr_trial": best_components["ppr"]["config"]["trial_id"],
                    "mf_trial": best_components["mf"]["config"]["trial_id"],
                }
            preds = get_predictions(data, scores, users, method)
            score = mapk(truth, preds)
            record = {
                "config": config,
                "config_hash": config_hash(config),
                "status": "complete",
                "map5": score,
                "seconds": time.perf_counter() - t0,
                "details": info,
                "provenance": provenance,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
            }
            if method in {"ppr", "mf"} and (
                method not in best_components or score > best_components[method]["map5"]
            ):
                best_components[method] = record
                component_scores[method] = scores.copy()
                save_predictions(output_dir / "predictions" / f"{method}.json", preds)
            json_write(output_dir / "trials" / f"{trial_id}.json", record)
            by_id[trial_id] = record
            print(
                f"{trial_id} {method:6s} MAP@5={score:.7f} seconds={record['seconds']:.2f}",
                flush=True,
            )
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            record = {
                "config": config,
                "config_hash": config_hash(config),
                "status": "failed",
                "error": str(exc),
                "seconds": time.perf_counter() - t0,
                "provenance": provenance,
            }
            json_write(output_dir / "trials" / f"{trial_id}.json", record)
            by_id[trial_id] = record
            print(f"{trial_id} FAILED: {exc}", flush=True)
    records = [by_id[c["trial_id"]] for c in plan if c["trial_id"] in by_id]
    successful = [r for r in records if r["status"] == "complete"]
    if not successful:
        raise RuntimeError("No experiment completed")
    best = max(successful, key=lambda r: (r["map5"], -int(r["config"]["trial_id"].split("_")[-1])))
    if baseline_score > best["map5"]:
        winner = {"method": "popularity"}
        best_score = baseline_score
    else:
        winner = best["config"]
        best_score = best["map5"]
    if winner["method"] == "hybrid":
        for component in ["ppr", "mf"]:
            if component not in component_scores:
                component_scores[component], _ = model_scores(
                    data, best_components[component]["config"]
                )
        best_scores = fuse(
            component_scores["ppr"],
            component_scores["mf"],
            data.items,
            data.pool,
            winner["weight_mf"],
            winner["kind"],
        )
        if winner["schedule_beta"]:
            best_scores *= schedule_ratio(data) ** winner["schedule_beta"]
        final_preds = get_predictions(data, best_scores, users, "hybrid")
    elif winner["method"] == "popularity":
        final_preds = baseline
    else:
        if winner["method"] in component_scores:
            best_scores = component_scores[winner["method"]]
        else:
            best_scores, _ = model_scores(data, winner)
        final_preds = get_predictions(data, best_scores, users, winner["method"])
    for method, record in best_components.items():
        if method not in component_scores:
            component_scores[method], _ = model_scores(data, record["config"])
        save_predictions(
            output_dir / "predictions" / f"{method}.json",
            get_predictions(data, component_scores[method], users, method),
        )
    save_predictions(output_dir / "predictions" / "selected.json", final_preds)
    summary = {
        "provenance": provenance,
        "git_commit_at_run": git_commit(),
        "planned_trials": 150,
        "completed_trials": len(successful),
        "failed_trials": len(records) - len(successful),
        "search_complete": len(successful) == 150,
        "cohort": cohort,
        "baseline_map5": baseline_score,
        "selected_config": winner,
        "selected_map5": best_score,
        "best_components": {
            k: {"config": v["config"], "map5": v["map5"]} for k, v in best_components.items()
        },
        "resumed_completed_trials": sum(r["status"] == "complete" for r in previous),
        "seconds_this_invocation": time.perf_counter() - started,
        "seconds_all_trial_records": sum(r["seconds"] for r in records),
        "validation_window": "history weeks 11-20 -> target weeks 21-30",
        "selection_caveat": "150 configurations use this holdout for selection; Kaggle private score is the external result",
    }
    json_write(output_dir / "summary.json", summary)
    with (output_dir / "trials.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["trial_id", "method", "status", "map5", "seconds", "config_json"]
        )
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "trial_id": r["config"]["trial_id"],
                    "method": r["config"]["method"],
                    "status": r["status"],
                    "map5": r.get("map5", ""),
                    "seconds": r["seconds"],
                    "config_json": json.dumps(r["config"], sort_keys=True),
                }
            )
    build_examples(cache_dir, output_dir, summary, truth)
    return summary


def build_examples(
    cache_dir: Path, output_dir: Path, summary: dict, truth: dict[int, list[int]]
) -> dict:
    data = load_dataset(cache_dir)
    activity = np.asarray(data.matrix.sum(1)).ravel()
    ids = np.array(list(truth), dtype=np.uint64)
    rows, known = user_lookup(data.users, ids)
    warm = ids[known]
    if not len(warm):
        picked = {"cold": int(ids[0]), "second": int(ids[min(1, len(ids) - 1)])}
    else:
        warm_rows = rows[known]
        values = activity[warm_rows]
        heavy = int(warm[np.argmin(np.abs(values - np.percentile(values, 90)))])
        light = int(warm[np.argmin(np.abs(values - np.percentile(values, 10)))])
        if light == heavy and len(warm) > 1:
            light = int(next(u for u in warm if int(u) != heavy))
        picked = {"heavy": heavy, "light": light}
        if (~known).any():
            picked["light"] = int(ids[~known][0])
    metadata = json.loads((cache_dir / "metadata.json").read_text())
    popularity = dict(zip(map(int, data.items), map(float, np.asarray(data.matrix.sum(0)).ravel())))
    methods = {}
    for name in ["popularity", "ppr", "mf", "selected"]:
        path = output_dir / "predictions" / f"{name}.json"
        if path.is_file():
            methods[name] = json.loads(path.read_text())
    out = {
        "choice_rule": "fixed 90th/10th history-activity percentiles; replace light with first cold user if available; ties ascending user id",
        "users": {},
        "data_kind": summary["provenance"]["data_kind"],
    }

    def item(show):
        return {
            "id": int(show),
            **metadata.get(str(show), {"title": "[metadata unavailable]", "category": ""}),
            "training_popularity": popularity.get(int(show), 0.0),
        }

    for name, user in picked.items():
        row, ok = user_lookup(data.users, np.array([user], dtype=np.uint64))
        history = []
        if ok[0]:
            vector = data.matrix[int(row[0])]
            history = [
                {**item(data.items[j]), "qualifying_airings": float(v)}
                for j, v in zip(vector.indices, vector.data)
            ]
            history.sort(key=lambda x: (-x["qualifying_airings"], x["id"]))
        seen = {v["id"] for v in history}
        entries = {}
        for method, lists in methods.items():
            rec = lists[str(user)]
            entries[method] = {
                "ap5": apk(truth[user], rec),
                "top5": [
                    {**item(s), "hit": s in truth[user], "in_history": s in seen} for s in rec
                ],
                "mean_training_popularity": float(np.mean([popularity.get(s, 0.0) for s in rec])),
            }
        out["users"][name] = {
            "user_id": str(user),
            "cold_start": not bool(ok[0]),
            "history": history[:8],
            "history_distinct_programmes": len(history),
            "actual_top5": [item(s) for s in truth[user]],
            "recommendations": entries,
        }
    json_write(output_dir / "examples.json", out)
    return out


def predict(
    cache_dir: Path, results_dir: Path, submission: Path, allow_fixture: bool = False
) -> dict:
    summary = json.loads((results_dir / "summary.json").read_text())
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    if not summary["search_complete"]:
        raise ValueError("Final prediction requires 150 completed experiments")
    if manifest["data_kind"] != "competition" and not allow_fixture:
        raise ValueError("Cannot create an exam submission from synthetic data")
    if (
        summary["provenance"]["data_fingerprint"] != manifest["fingerprint"]
        or summary["provenance"]["source_fingerprint"] != source_fingerprint()
    ):
        raise ValueError(
            "Data or implementation changed after tuning; rerun in a fresh results directory"
        )
    data = load_dataset(cache_dir, final=True)
    users = np.load(cache_dir / "template_users.npy", allow_pickle=False)
    selected = summary["selected_config"]
    method = selected["method"]
    t0 = time.perf_counter()
    info = {}
    if method == "popularity":
        recommendations = {int(user): fallback_list(data)[:5] for user in users}
    else:
        if method == "hybrid":
            ppr, ip = model_scores(data, summary["best_components"]["ppr"]["config"])
            mf, im = model_scores(data, summary["best_components"]["mf"]["config"])
            scores = fuse(ppr, mf, data.items, data.pool, selected["weight_mf"], selected["kind"])
            if selected["schedule_beta"]:
                scores *= schedule_ratio(data) ** selected["schedule_beta"]
            info = {"ppr": ip, "mf": im}
        else:
            scores, info = model_scores(data, selected)
        recommendations = get_predictions(data, scores, users, method)
    submission.parent.mkdir(parents=True, exist_ok=True)
    pool = set(map(int, data.pool))
    with submission.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "tv_show_id"])
        for user in users:
            rec = recommendations[int(user)]
            if len(rec) != 5 or len(set(rec)) != 5 or not set(rec) <= pool or 0 in rec:
                raise AssertionError("Invalid final recommendation list")
            writer.writerow([str(int(user)), " ".join(map(str, rec))])
    _, warm = user_lookup(data.users, users)
    record = {
        "data_kind": manifest["data_kind"],
        "submission_filename": submission.name,
        "sha256": sha256(submission),
        "users": len(users),
        "cold_start_users": int((~warm).sum()),
        "candidate_programmes": len(pool),
        "selected_config": selected,
        "seconds": time.perf_counter() - t0,
        "model_details": info,
        "provenance": summary["provenance"],
        "kaggle_status": "not_submitted",
    }
    json_write(results_dir / "final_prediction.json", record)
    return record
