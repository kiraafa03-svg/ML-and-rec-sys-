import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from exam.data import prepare
from exam.experiments import predict, run_search
from exam.session import copy_native_session, inspect_session
from exam.tests.fixture import make_fixture
from exam.validation import verify_results
from scripts.record_kaggle_score import parse_returned_score


class WorkflowTests(unittest.TestCase):
    def test_all_150_configs_resume_and_deterministic_csv(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            make_fixture(root / "data")
            prepare(root / "data", root / "cache", fixture=True)
            with self.assertRaises(ValueError):
                run_search(root / "cache", root / "blocked")
            first = run_search(root / "cache", root / "results", limit=2, allow_fixture=True)
            self.assertEqual(first["completed_trials"], 2)
            with self.assertRaises(ValueError):
                predict(
                    root / "cache", root / "results", root / "not_final.csv", allow_fixture=True
                )
            full = run_search(root / "cache", root / "results", resume=True, allow_fixture=True)
            self.assertTrue(full["search_complete"])
            self.assertEqual(full["completed_trials"], 150)
            a = predict(root / "cache", root / "results", root / "a.csv", allow_fixture=True)
            b = predict(root / "cache", root / "results", root / "b.csv", allow_fixture=True)
            self.assertEqual(a["sha256"], b["sha256"])
            verified = verify_results(root / "cache", root / "results", root / "b.csv")
            self.assertTrue(verified["passed"])
            self.assertEqual(verified["data_kind"], "synthetic_fixture")
            with self.assertRaises(ValueError):
                predict(root / "cache", root / "results", root / "blocked.csv")
            trial = root / "results/trials/trial_001.json"
            value = json.loads(trial.read_text())
            value["config"]["alpha"] = 0.123
            trial.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                run_search(root / "cache", root / "results", resume=True, allow_fixture=True)

    def test_kaggle_score_parser_uses_returned_private_score(self):
        text = "ref,status,publicScore,privateScore\n123,COMPLETE,0.4,0.3\n"
        self.assertEqual(parse_returned_score(text, 123)["private_score"], 0.3)
        with self.assertRaises(ValueError):
            parse_returned_score(text, 124)
        with self.assertRaises(ValueError):
            parse_returned_score("ref,status,privateScore\n123,PENDING,\n", 123)

    def test_session_copy_is_exact_and_link_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # This is explicitly a parser fixture, not a real session export.
            raw = b'{"type":"function_call","name":"synthetic_parser_fixture"}\n{"type":"function_call_output","output":"synthetic_parser_fixture"}\n'
            source = root / "fixture.jsonl"
            source.write_bytes(raw)
            out = root / "copy.jsonl"
            info = copy_native_session(source, out)
            self.assertTrue(info["byte_identical_copy"])
            self.assertEqual(out.read_bytes(), raw)
            source.write_text('{"session_url":"https://example.invalid/not-a-session"}')
            with self.assertRaises(ValueError):
                inspect_session(source)


if __name__ == "__main__":
    unittest.main()
