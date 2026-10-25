"""Small SYNTHETIC data for software tests; never a Kaggle experiment."""

import csv
from datetime import datetime, timedelta
from pathlib import Path

from exam.data import FILES


def make_fixture(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    headers = [
        "channel_id",
        "channel_title",
        "start_time",
        "duration",
        "tv_show_title",
        "tv_show_id",
        "tv_show_category",
    ]
    windows = [
        ("schedule_11_20", datetime(2020, 3, 9)),
        ("schedule_21_30", datetime(2020, 5, 18)),
        ("schedule_31_42", datetime(2020, 7, 27)),
    ]
    all_views = []
    for source, base in windows:
        rows = []
        for repetition in range(2):
            for item in range(1, 9):
                start = base + timedelta(days=repetition, hours=item)
                rows.append(
                    [
                        1,
                        "SYNTHETIC",
                        start.strftime("%d.%m.%Y %H:%M:%S"),
                        600,
                        f"SYNTHETIC show {item}",
                        item,
                        "Fixture",
                    ]
                )
                if source != "schedule_31_42":
                    for user in [9007199254740993, 9007199254740995, 18446744073709551614]:
                        if (user % 3 + item) % 4 != 0:
                            all_views.append(
                                [
                                    user,
                                    1,
                                    start.strftime("%Y-%m-%d %H:%M:%S"),
                                    (start + timedelta(seconds=600)).strftime("%Y-%m-%d %H:%M:%S"),
                                    600,
                                ]
                            )
        # Unlabelled airtime must not become item 0.
        rows.append([2, "SYNTHETIC", base.strftime("%d.%m.%Y %H:%M:%S"), 600, "Unlabelled", 0, ""])
        with (directory / FILES[source]).open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
    with (directory / FILES["views"]).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "vsetv_id", "start_time", "stop_time", "duraton"])
        writer.writerows(all_views)
    with (directory / FILES["template"]).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "tv_show_id"])
        writer.writerows(
            [
                [9007199254740993, "1 2 3 4 5"],
                [9007199254740994, "1 2 3 4 5"],
                [18446744073709551615, "1 2 3 4 5"],
            ]
        )
