from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from exam.data import (
    AIR_DTYPE,
    COUNT_DTYPE,
    VIEW_DTYPE,
    build_counts,
    clip_views,
    deduplicate_airings,
    ground_truth,
    interval_join,
    load_dataset,
    prepare,
    week_start,
)
from exam.experiments import trial_plan
from exam.metrics import apk, mapk
from exam.models import (
    WeightedALS,
    fallback_list,
    fuse,
    ppr_graph,
    ppr_scores,
    rank_top5,
    schedule_ratio,
)
from exam.tests.fixture import make_fixture
from exam.validation import independent_ap, verify_submission


class MetricTests(unittest.TestCase):
    def test_hand_ap(self):
        self.assertAlmostEqual(apk({1, 2, 3, 4, 5}, [1, 2, 9, 3, 4]), 0.71)

    def test_duplicate_predictions_do_not_add_hits(self):
        self.assertAlmostEqual(apk({1, 2, 3}, [1, 1, 2, 9, 3]), (1 + 2 / 3 + 3 / 5) / 3)

    def test_empty_truth(self):
        self.assertEqual(apk([], [1, 2, 3, 4, 5]), 0)

    def test_missing_users_raise(self):
        with self.assertRaises(ValueError):
            mapk({10: [1]}, {})

    def test_empty_cohort_raises(self):
        with self.assertRaises(ValueError):
            mapk({}, {})

    def test_independent_oracle_random_cases(self):
        rng = np.random.default_rng(7)
        for _ in range(100):
            truth = rng.choice(12, 5, replace=False).tolist()
            pred = rng.integers(0, 12, size=5).tolist()
            self.assertAlmostEqual(apk(truth, pred), independent_ap(truth, pred))


class JoinTests(unittest.TestCase):
    def collect(self, v, a):
        rows = []
        for users, airings, overlaps in interval_join(v, a, batch_size=2):
            rows.extend(zip(map(int, users), map(int, airings), map(int, overlaps)))
        return sorted(rows)

    def test_empty_join(self):
        self.assertEqual(
            self.collect(
                np.array([(1, 1, 200, 300)], dtype=VIEW_DTYPE),
                np.array([(1, 0, 100, 1)], dtype=AIR_DTYPE),
            ),
            [],
        )

    def test_half_open_endpoints(self):
        a = np.array([(1, 0, 100, 1)], dtype=AIR_DTYPE)
        v = np.array([(1, 1, 100, 200), (2, 1, -100, 0)], dtype=VIEW_DTYPE)
        self.assertEqual(self.collect(v, a), [])

    def test_nonmonotone_ends(self):
        a = np.array([(1, 0, 100, 1), (1, 10, 10, 2)], dtype=AIR_DTYPE)
        v = np.array([(1, 1, 30, 40)], dtype=VIEW_DTYPE)
        self.assertEqual(self.collect(v, a), [(1, 0, 10)])

    def test_random_join_against_brute_force(self):
        rng = np.random.default_rng(42)
        for _ in range(40):
            a = np.array(
                [
                    (
                        int(rng.integers(1, 4)),
                        int(rng.integers(0, 200)),
                        int(rng.integers(1, 100)),
                        j + 1,
                    )
                    for j in range(30)
                ],
                dtype=AIR_DTYPE,
            )
            v = np.array(
                [
                    (u, int(rng.integers(1, 4)), int(t), int(t + rng.integers(1, 100)))
                    for u, t in enumerate(rng.integers(0, 200, size=20))
                ],
                dtype=VIEW_DTYPE,
            )
            expected = []
            for row in v:
                for j, air in enumerate(a):
                    ov = min(row["stop"], air["start"] + air["duration"]) - max(
                        row["start"], air["start"]
                    )
                    if row["channel"] == air["channel"] and ov > 0:
                        expected.append((int(row["user"]), j, int(ov)))
            self.assertEqual(self.collect(v, a), sorted(expected))

    def test_merge_duplicate_and_adjacent_views(self):
        v = np.array(
            [(1, 1, 0, 60), (1, 1, 0, 60), (1, 1, 60, 80), (1, 2, 0, 20)], dtype=VIEW_DTYPE
        )
        out = clip_views(v, 0, 100)
        self.assertEqual(out.tolist(), [(1, 1, 0, 80), (1, 2, 0, 20)])

    def test_future_view_does_not_enter_training(self):
        cutoff = week_start(21)
        a = np.array([(1, cutoff - 1, 10, 1)], dtype=AIR_DTYPE)
        v = np.array([(1, 1, cutoff, cutoff + 9)], dtype=VIEW_DTYPE)
        counts, _, stats = build_counts(v, a, 11, 21)
        self.assertEqual(len(counts), 0)
        self.assertEqual(stats["qualifying_airings"], 0)

    def test_duplicate_sum_and_union_are_explicit(self):
        start = week_start(11)
        a = np.array([(1, start, 100, 1)], dtype=AIR_DTYPE)
        v = np.array([(1, 1, start, start + 60), (1, 1, start, start + 60)], dtype=VIEW_DTYPE)
        union, _, _ = build_counts(v, a, 11, 21, overlap_policy="union")
        summed, _, stats = build_counts(v, a, 11, 21, overlap_policy="sum")
        self.assertEqual(len(union), 0)
        self.assertEqual(len(summed), 1)
        self.assertEqual(stats["screen_over_one"], 1)

    def test_exact_dedup_keeps_conflicts(self):
        a = np.array([(1, 0, 100, 1), (1, 0, 100, 1), (1, 0, 100, 2)], dtype=AIR_DTYPE)
        out, stats = deduplicate_airings(a)
        self.assertEqual(len(out), 2)
        self.assertEqual(stats["exact_duplicates_dropped"], 1)


class ModelTests(unittest.TestCase):
    def test_ppr_matches_linear_system_with_dangling(self):
        graph = sparse.csr_matrix([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        e = np.array([0.2, 0.3, 0.5])
        restart = sparse.csr_matrix([e])
        scores, info = ppr_scores(graph, restart, alpha=0.85, tolerance=1e-11, max_iter=500)
        effective = graph.toarray()
        effective[2] = e
        expected = spsolve(sparse.eye(3) - 0.85 * sparse.csr_matrix(effective.T), 0.15 * e)
        np.testing.assert_allclose(scores[0], expected, atol=3e-8)
        self.assertTrue(info["converged"])
        self.assertAlmostEqual(float(scores.sum()), 1.0, places=6)

    def test_ppr_cap_is_not_convergence(self):
        graph = sparse.csr_matrix([[0.0, 1.0], [1.0, 0.0]])
        _, info = ppr_scores(graph, sparse.csr_matrix([[1.0, 0.0]]), alpha=0.95, max_iter=1)
        self.assertFalse(info["converged"])

    def test_graph_rows_stochastic(self):
        graph, _ = ppr_graph(sparse.csr_matrix([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]), topk=2)
        np.testing.assert_allclose(np.asarray(graph.sum(1)).ravel(), 1)

    def test_als_objective_matches_dense_oracle(self):
        R = sparse.csr_matrix([[1.0, 0.0, 3.0], [0.0, 2.0, 1.0]])
        model = WeightedALS(
            factors=3, iterations=4, confidence=3, regularization=0.05, target="log"
        ).fit(R)
        dense = R.toarray()
        C = 1 + 3 * np.log1p(dense)
        P = np.log1p(dense)
        pred = model.user_factors @ model.item_factors.T
        expected = float(
            np.sum(C * (P - pred) ** 2)
            + 0.05 * (np.sum(model.user_factors**2) + np.sum(model.item_factors**2))
        )
        self.assertAlmostEqual(model.objective(R), expected, places=10)

    def test_als_normal_equation_and_descent(self):
        R = sparse.csr_matrix([[1.0, 0.0, 3.0], [0.0, 2.0, 1.0]])
        model = WeightedALS(factors=3, iterations=5).fit(R)
        trace = model.info["objective"]
        self.assertLessEqual(trace[-1]["value"], trace[0]["value"] + 1e-10)
        X = model._step(R, model.item_factors)
        Y = model.item_factors
        c = 1 + model.confidence * np.log1p(R.toarray()[0])
        A = Y.T @ (c[:, None] * Y) + model.regularization * np.eye(3)
        b = Y.T @ (c * (R.toarray()[0] > 0))
        np.testing.assert_allclose(A @ X[0], b, atol=1e-11)

    def test_als_seed_reproducibility(self):
        R = sparse.csr_matrix([[1.0, 0.0, 3.0], [0.0, 2.0, 1.0]])
        first = WeightedALS(factors=3, iterations=3).fit(R).scores()
        second = WeightedALS(factors=3, iterations=3).fit(R).scores()
        np.testing.assert_array_equal(first, second)

    def test_fusion_uses_same_candidate_mask(self):
        ppr = np.array([[0.1, 0.9, 0.2]], dtype=np.float32)
        mf = np.array([[1.0, 100.0, 3.0]], dtype=np.float32)
        out = fuse(ppr, mf, np.array([1, 2, 3]), np.array([1, 3]), weight_mf=0.5)
        self.assertEqual(out[0, 1], 0)
        self.assertGreater(out[0, 2], out[0, 0])


class DataIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        make_fixture(self.root / "data")
        self.manifest = prepare(self.root / "data", self.root / "cache", fixture=True)
        self.data = load_dataset(self.root / "cache")

    def tearDown(self):
        self.temp.cleanup()

    def test_synthetic_provenance(self):
        self.assertEqual(self.manifest["data_kind"], "synthetic_fixture")

    def test_template_uint64_round_trip_and_cold_start(self):
        users = np.load(self.root / "cache/template_users.npy")
        scores = np.zeros(self.data.matrix.shape, dtype=np.float32)
        lists = rank_top5(scores, self.data, users, min_score=0.0)
        self.assertEqual(lists[int(users[-1])], fallback_list(self.data)[:5])
        self.assertEqual(int(users[-1]), 18446744073709551615)
        self.assertEqual(len(set(lists[int(users[1])])), 5)

    def test_no_item_zero(self):
        self.assertNotIn(0, self.data.items)

    def test_schedule_ratio_finite(self):
        self.assertTrue(np.isfinite(schedule_ratio(self.data)).all())

    def test_truth_tie_rule_and_exclusions(self):
        c = np.array([(1, i, 21, 1) for i in range(1, 7)] + [(2, 1, 21, 1)], dtype=COUNT_DTYPE)
        truth, stats = ground_truth(c)
        self.assertEqual(truth, {1: [1, 2, 3, 4, 5]})
        self.assertEqual(stats["users_below_minimum_excluded"], 1)
        self.assertEqual(stats["top5_boundary_ties"], 1)

    def test_real_ingestion_rejects_toy_rows(self):
        with self.assertRaisesRegex(ValueError, "Row counts differ"):
            prepare(self.root / "data", self.root / "rejected")

    def test_submission_checks_exact_user_order(self):
        path = self.root / "final.csv"
        users = np.load(self.root / "cache/template_users.npy")
        path.write_text(
            "user_id,tv_show_id\n" + "\n".join(f"{int(u)},1 2 3 4 5" for u in users) + "\n"
        )
        self.assertTrue(verify_submission(path, users, self.data.pool)["passed"])
        with self.assertRaises(ValueError):
            verify_submission(path, users[::-1], self.data.pool)

    def test_exactly_150_distinct_configs(self):
        plan = trial_plan()
        self.assertEqual(len(plan), 150)
        self.assertEqual(
            {m: sum(c["method"] == m for c in plan) for m in ["ppr", "mf", "hybrid"]},
            {"ppr": 40, "mf": 70, "hybrid": 40},
        )


if __name__ == "__main__":
    unittest.main()
