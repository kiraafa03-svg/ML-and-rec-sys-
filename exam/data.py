"""Streaming CSV ingestion and leakage-safe, exact interval matching.

Times are naive wall-clock seconds, represented by UTC arithmetic without an
implicit machine-timezone conversion. All intervals are half-open [start, stop).
User IDs stay uint64 or Python integers; they never pass through float64.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy import sparse

EPOCH = datetime(1970, 1, 1)
WEEK0 = 1583712000  # 2020-03-09, Monday of ISO week 11
WEEK = 604800
FORMAT_VERSION = 1
FILES = {
    "views": "dataset11-30.csv",
    "schedule_11_20": "export_arh_11-20-final.csv",
    "schedule_21_30": "export_arh_21-30-final.csv",
    "schedule_31_42": "export_arh_31-42-final.csv",
    "template": "submission.csv",
}
VIEW_COLUMNS = {"user_id", "vsetv_id", "start_time", "stop_time", "duraton"}
SCHEDULE_COLUMNS = {
    "channel_id",
    "channel_title",
    "start_time",
    "duration",
    "tv_show_id",
    "tv_show_title",
    "tv_show_category",
}
VIEW_DTYPE = np.dtype([("user", "<u8"), ("channel", "<i4"), ("start", "<i8"), ("stop", "<i8")])
AIR_DTYPE = np.dtype([("channel", "<i4"), ("start", "<i8"), ("duration", "<i8"), ("show", "<i8")])
COUNT_DTYPE = np.dtype([("user", "<u8"), ("show", "<i8"), ("week", "<i2"), ("count", "<i4")])
CHANNEL_DTYPE = np.dtype([("user", "<u8"), ("channel", "<i4"), ("seconds", "<i8")])


def week_start(week: int) -> int:
    return WEEK0 + (week - 11) * WEEK


def seconds(value: str, schedule: bool = False) -> int:
    value = value.strip()
    date = (
        datetime.strptime(value, "%d.%m.%Y %H:%M:%S") if schedule else datetime.fromisoformat(value)
    )
    if date.tzinfo is not None:
        raise ValueError(
            "Expected naive source times; a timezone offset needs an explicit alignment decision"
        )
    return int((date - EPOCH).total_seconds())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temp.replace(path)


def reader(path: Path, required: set[str]):
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = csv.DictReader(f)
        missing = required.difference(rows.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
        yield from rows


def read_template(path: Path) -> np.ndarray:
    users = [int(row["user_id"]) for row in reader(path, {"user_id", "tv_show_id"})]
    if len(set(users)) != len(users):
        raise ValueError("Submission template contains duplicate users")
    if not users:
        raise ValueError("Submission template is empty")
    return np.array(users, dtype=np.uint64)


def preflight(data_dir: Path) -> dict:
    paths = {name: data_dir / filename for name, filename in FILES.items()}
    missing = [p.name for p in paths.values() if not p.is_file()]
    return {
        "ready": not missing,
        "data_dir": str(data_dir.resolve()),
        "missing": missing,
        "present": {
            name: {"filename": p.name, "size_bytes": p.stat().st_size}
            for name, p in paths.items()
            if p.is_file()
        },
    }


def _chunked_array(rows, dtype: np.dtype, chunk_size: int = 100_000) -> np.ndarray:
    chunks, block = [], []
    for row in rows:
        block.append(row)
        if len(block) == chunk_size:
            chunks.append(np.array(block, dtype=dtype))
            block = []
    if block:
        chunks.append(np.array(block, dtype=dtype))
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=dtype)


def read_views(path: Path) -> tuple[np.ndarray, dict]:
    stats = Counter()

    def rows():
        for number, r in enumerate(reader(path, VIEW_COLUMNS), 2):
            stats["rows"] += 1
            try:
                user, channel = int(r["user_id"]), int(r["vsetv_id"])
                start, stop = seconds(r["start_time"]), seconds(r["stop_time"])
                duration = int(r["duraton"])
                if not 0 <= user <= np.iinfo(np.uint64).max:
                    raise ValueError("user_id outside uint64")
                if not np.iinfo(np.int32).min <= channel <= np.iinfo(np.int32).max:
                    raise ValueError("channel_id outside int32")
            except (ValueError, OverflowError) as exc:
                raise ValueError(f"{path.name}:{number}: {exc}") from exc
            if stop <= start:
                stats["nonpositive_intervals_dropped"] += 1
                continue
            stats["duration_mismatch"] += int(duration != stop - start)
            yield user, channel, start, stop

    views = _chunked_array(rows(), VIEW_DTYPE)
    stats["valid_rows"] = len(views)
    if len(views):
        durations = views["stop"] - views["start"]
        stats.update(
            {
                "unique_users": len(np.unique(views["user"])),
                "max_view_seconds": int(durations.max()),
                "median_view_seconds": float(np.median(durations)),
            }
        )
    return views, dict(stats)


def read_schedule(path: Path) -> tuple[np.ndarray, dict[int, dict], dict]:
    stats, titles, categories = Counter(), {}, {}

    def rows():
        for number, r in enumerate(reader(path, SCHEDULE_COLUMNS), 2):
            stats["rows"] += 1
            try:
                show = int(r["tv_show_id"] or 0)
                if show == 0:
                    stats["unlabelled_rows"] += 1
                    continue
                if show < 0:
                    raise ValueError("negative programme id")
                start, duration = seconds(r["start_time"], True), int(r["duration"])
                channel = int(r["channel_id"])
            except (ValueError, OverflowError) as exc:
                raise ValueError(f"{path.name}:{number}: {exc}") from exc
            if duration <= 0:
                stats["nonpositive_duration_dropped"] += 1
                continue
            titles.setdefault(show, Counter())[r["tv_show_title"]] += 1
            categories.setdefault(show, Counter())[r["tv_show_category"]] += 1
            yield channel, start, duration, show

    airings = _chunked_array(rows(), AIR_DTYPE)

    # Lexical tie-breaking makes metadata stable even if the input row order changes.
    def mode(counts):
        return min(counts, key=lambda value: (-counts[value], value))

    metadata = {
        show: {"title": mode(titles[show]), "category": mode(categories[show])} for show in titles
    }
    stats["labelled_rows"] = len(airings)
    stats["distinct_programmes"] = len(metadata)
    return airings, metadata, dict(stats)


def deduplicate_airings(airings: np.ndarray) -> tuple[np.ndarray, dict]:
    unique = np.unique(airings)  # exact channel/start/duration/show duplicates only
    unique = unique[np.lexsort((unique["show"], unique["start"], unique["channel"]))]
    if len(unique):
        same = (unique["channel"][1:] == unique["channel"][:-1]) & (
            unique["start"][1:] == unique["start"][:-1]
        )
        conflicts = int(same.sum())
    else:
        conflicts = 0
    return unique, {
        "exact_duplicates_dropped": len(airings) - len(unique),
        "remaining_same_channel_start_rows": conflicts,
    }


def clip_views(views: np.ndarray, start: int, stop: int, policy: str = "union") -> np.ndarray:
    if stop <= start:
        raise ValueError("Invalid time window")
    if policy not in {"union", "sum"}:
        raise ValueError("Unknown overlap policy")
    mask = (views["start"] < stop) & (views["stop"] > start)
    v = views[mask].copy()
    v["start"] = np.maximum(v["start"], start)
    v["stop"] = np.minimum(v["stop"], stop)
    if policy == "sum" or not len(v):
        return v
    order = np.lexsort((v["stop"], v["start"], v["channel"], v["user"]))
    v = v[order]
    # Merge overlapping/adjacent intervals within each (user, channel).
    output = np.empty(len(v), dtype=VIEW_DTYPE)
    n = 0
    for row in v:
        if (
            n
            and row["user"] == output[n - 1]["user"]
            and row["channel"] == output[n - 1]["channel"]
            and row["start"] <= output[n - 1]["stop"]
        ):
            output[n - 1]["stop"] = max(output[n - 1]["stop"], row["stop"])
        else:
            output[n] = row
            n += 1
    return output[:n]


def interval_join(views: np.ndarray, airings: np.ndarray, batch_size: int = 50_000):
    """All positive overlaps, robust to nested/overlapping schedule intervals.

    prefix_max(end), unlike end itself, is sorted. Its search bound can include
    extra candidates; the exact overlap filter then removes them.
    Yields bounded batches of (user_id, airing_index, overlap_seconds).
    """
    for channel in np.intersect1d(np.unique(views["channel"]), np.unique(airings["channel"])):
        ai = np.flatnonzero(airings["channel"] == channel)
        ai = ai[np.argsort(airings["start"][ai], kind="stable")]
        starts = airings["start"][ai]
        ends = starts + airings["duration"][ai]
        prefix_end = np.maximum.accumulate(ends)
        vi = np.flatnonzero(views["channel"] == channel)
        for first in range(0, len(vi), batch_size):
            vb = views[vi[first : first + batch_size]]
            low = np.searchsorted(prefix_end, vb["start"], side="right")
            high = np.searchsorted(starts, vb["stop"], side="left")
            count = np.maximum(high - low, 0)
            # A single very long view can overlap the whole channel grid. Chunk
            # views adaptively, before constructing repeated candidate arrays.
            if int(count.sum()) > 2_000_000 and len(vb) > 1:
                for out in interval_join(vb, airings[ai], max(1, len(vb) // 8)):
                    u, a, overlap = out
                    yield u, ai[a], overlap
                continue
            total = int(count.sum())
            if not total:
                continue
            rep = np.repeat(np.arange(len(vb)), count)
            offset = np.arange(total) - np.repeat(np.cumsum(count) - count, count)
            local = low[rep] + offset
            overlap = np.minimum(vb["stop"][rep], ends[local]) - np.maximum(
                vb["start"][rep], starts[local]
            )
            valid = overlap > 0
            if valid.any():
                yield vb["user"][rep][valid], ai[local[valid]], overlap[valid]


def build_counts(
    views: np.ndarray,
    airings: np.ndarray,
    first_week: int,
    stop_week: int,
    min_fraction: float = 0.8,
    overlap_policy: str = "union",
):
    if not 0 < min_fraction <= 1:
        raise ValueError("min_fraction must be in (0, 1]")
    left, right = week_start(first_week), week_start(stop_week)
    v = clip_views(views, left, right, overlap_policy)
    a = airings[(airings["start"] >= left) & (airings["start"] < right)]
    pair_dtype = np.dtype([("user", "<u8"), ("airing", "<i8")])
    pair_blocks, weights = [], []
    for user, airing, overlap in interval_join(v, a):
        pairs = np.empty(len(user), dtype=pair_dtype)
        pairs["user"], pairs["airing"] = user, airing
        pair_blocks.append(pairs)
        weights.append(overlap)
    stats = {
        "first_week": first_week,
        "stop_week_exclusive": stop_week,
        "input_view_rows": len(v),
        "eligible_airings": len(a),
        "overlap_policy": overlap_policy,
        "min_fraction": min_fraction,
        "raw_user_airing_overlaps": int(sum(map(len, pair_blocks))),
    }
    if pair_blocks:
        pairs = np.concatenate(pair_blocks)
        pairs, inverse = np.unique(pairs, return_inverse=True)
        seconds_total = np.bincount(inverse, weights=np.concatenate(weights))
        duration = a["duration"][pairs["airing"]]
        screen = seconds_total / duration
        if overlap_policy == "union" and np.any(screen > 1 + 1e-10):
            raise AssertionError("Merged intervals exceeded programme length")
        stats.update(
            {"user_airing_pairs": len(pairs), "screen_over_one": int((screen > 1 + 1e-10).sum())}
        )
        qualifies = seconds_total + 1e-9 >= min_fraction * duration
        pairs = pairs[qualifies]
        base = np.empty(len(pairs), dtype=COUNT_DTYPE)
        base["user"] = pairs["user"]
        base["show"] = a["show"][pairs["airing"]]
        base["week"] = ((a["start"][pairs["airing"]] - WEEK0) // WEEK + 11).astype(np.int16)
        key = base[["user", "show", "week"]]
        _, ix, n = np.unique(key, return_index=True, return_counts=True)
        counts = base[ix].copy()
        counts["count"] = n
        stats["qualifying_airings"] = len(pairs)
    else:
        counts = np.empty(0, dtype=COUNT_DTYPE)
        stats.update({"user_airing_pairs": 0, "screen_over_one": 0, "qualifying_airings": 0})
    # Context can use unlabelled airtime, but only clipped training views.
    if len(v):
        keys = v[["user", "channel"]]
        _, ix, inv = np.unique(keys, return_index=True, return_inverse=True)
        channel_context = np.empty(len(ix), dtype=CHANNEL_DTYPE)
        channel_context["user"], channel_context["channel"] = v["user"][ix], v["channel"][ix]
        channel_context["seconds"] = np.bincount(inv, weights=v["stop"] - v["start"]).astype(
            np.int64
        )
    else:
        channel_context = np.empty(0, dtype=CHANNEL_DTYPE)
    return counts, channel_context, stats


def prepare(
    data_dir: Path,
    cache_dir: Path,
    min_fraction: float = 0.8,
    overlap_policy: str = "union",
    fixture: bool = False,
) -> dict:
    status = preflight(data_dir)
    if not status["ready"]:
        raise FileNotFoundError("Missing competition data: " + ", ".join(status["missing"]))
    t0 = time.perf_counter()
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": FORMAT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_kind": "synthetic_fixture" if fixture else "competition",
        "files": {
            key: {
                "filename": name,
                "sha256": sha256(data_dir / name),
                "size_bytes": (data_dir / name).stat().st_size,
            }
            for key, name in FILES.items()
        },
        "configuration": {"min_fraction": min_fraction, "overlap_policy": overlap_policy},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    views, view_stats = read_views(data_dir / FILES["views"])
    parts, metadata, schedule_stats = [], {}, {}
    for key in ("schedule_11_20", "schedule_21_30", "schedule_31_42"):
        a, meta, stats = read_schedule(data_dir / FILES[key])
        parts.append(a)
        schedule_stats[key] = stats
        # Earlier metadata wins. Future titles/categories are for display only;
        # neither MF nor PPR receives metadata as a feature.
        for show, value in meta.items():
            metadata.setdefault(show, value)
    airings, dedup = deduplicate_airings(np.concatenate(parts))
    template = read_template(data_dir / FILES["template"])
    expected = {
        "views": 5_821_987,
        "schedule_11_20": 1_074_235,
        "schedule_21_30": 1_074_394,
        "schedule_31_42": 1_295_378,
        "template": 1260,
    }
    actual = {
        "views": view_stats["rows"],
        "template": len(template),
        **{key: value["rows"] for key, value in schedule_stats.items()},
    }
    if not fixture and actual != expected:
        raise ValueError(
            f"Row counts differ from the exam specification: {actual}. "
            "Check the data version. --fixture is exclusively for labelled synthetic tests."
        )
    np.save(cache_dir / "airings.npy", airings, allow_pickle=False)
    np.save(cache_dir / "template_users.npy", template, allow_pickle=False)
    json_write(cache_dir / "metadata.json", metadata)
    windows = {"train": (11, 21), "validation": (21, 31), "full_train": (11, 31)}
    summaries = {}
    for name, (first, stop) in windows.items():
        counts, channels, stats = build_counts(
            views, airings, first, stop, min_fraction, overlap_policy
        )
        np.save(cache_dir / f"{name}_counts.npy", counts, allow_pickle=False)
        np.save(cache_dir / f"{name}_channels.npy", channels, allow_pickle=False)
        summaries[name] = stats
    manifest.update(
        {
            "views": view_stats,
            "schedules": schedule_stats,
            "schedule_deduplication": dedup,
            "splits": summaries,
            "template_users": len(template),
            "template_users_absent_from_view_log": int(
                (~np.isin(template, np.unique(views["user"]))).sum()
            ),
            "seconds": time.perf_counter() - t0,
        }
    )
    manifest["cache_sha256"] = {p.name: sha256(p) for p in sorted(cache_dir.glob("*.npy"))}
    manifest["cache_sha256"]["metadata.json"] = sha256(cache_dir / "metadata.json")
    # Fingerprint covers the raw files and preprocessing decisions, not wall time.
    source = json.dumps(
        {
            "files": manifest["files"],
            "configuration": manifest["configuration"],
            "data_kind": manifest["data_kind"],
            "format_version": FORMAT_VERSION,
        },
        sort_keys=True,
    ).encode()
    manifest["fingerprint"] = hashlib.sha256(source).hexdigest()
    json_write(cache_dir / "manifest.json", manifest)
    return manifest


@dataclass
class Dataset:
    users: np.ndarray
    items: np.ndarray
    counts: np.ndarray
    matrix: sparse.csr_matrix
    pool: np.ndarray
    channels: np.ndarray
    airings: np.ndarray
    history_weeks: tuple[int, int]
    target_weeks: tuple[int, int]

    def weighted_matrix(self, decay: float = 0) -> sparse.csr_matrix:
        if decay < 0:
            raise ValueError("Recency decay must be nonnegative")
        c = self.counts
        weights = c["count"] * np.exp(-decay * (self.history_weeks[1] - 1 - c["week"]))
        return sparse.csr_matrix(
            (
                weights.astype(np.float64),
                (np.searchsorted(self.users, c["user"]), np.searchsorted(self.items, c["show"])),
            ),
            shape=(len(self.users), len(self.items)),
        )


def load_dataset(cache_dir: Path, final: bool = False) -> Dataset:
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    for name, digest in manifest.get("cache_sha256", {}).items():
        if sha256(cache_dir / name) != digest:
            raise ValueError(f"Prepared cache was modified: {name}")
    name, hist, target = (
        ("full_train", (11, 31), (31, 43)) if final else ("train", (11, 21), (21, 31))
    )
    counts = np.load(cache_dir / f"{name}_counts.npy", allow_pickle=False)
    users, items = np.unique(counts["user"]), np.unique(counts["show"])
    matrix = sparse.csr_matrix(
        (
            counts["count"].astype(np.float64),
            (np.searchsorted(users, counts["user"]), np.searchsorted(items, counts["show"])),
        ),
        shape=(len(users), len(items)),
    )
    airings = np.load(cache_dir / "airings.npy", allow_pickle=False)
    mask = (airings["start"] >= week_start(target[0])) & (airings["start"] < week_start(target[1]))
    pool = np.unique(airings["show"][mask])
    if len(pool) < 5:
        raise ValueError("Fewer than five labelled candidate programmes in the target schedule")
    return Dataset(
        users,
        items,
        counts,
        matrix,
        pool,
        np.load(cache_dir / f"{name}_channels.npy", allow_pickle=False),
        airings,
        hist,
        target,
    )


def ground_truth(counts: np.ndarray, min_items: int = 5) -> tuple[dict[int, list[int]], dict]:
    if min_items < 1:
        raise ValueError("min_items must be positive")
    totals = {}
    for row in counts:
        totals.setdefault(int(row["user"]), Counter())[int(row["show"])] += int(row["count"])
    truth, ties = {}, 0
    for user, values in sorted(totals.items()):
        order = sorted(values, key=lambda item: (-values[item], item))
        if len(order) >= min_items:
            truth[user] = order[:5]
            if len(order) > 5 and values[order[4]] == values[order[5]]:
                ties += 1
    return truth, {
        "target_users_with_qualifying_views": len(totals),
        "evaluable_users": len(truth),
        "users_below_minimum_excluded": len(totals) - len(truth),
        "minimum_distinct_items": min_items,
        "top5_boundary_ties": ties,
        "tie_rule": "ascending programme id; host tie rule unpublished",
    }
