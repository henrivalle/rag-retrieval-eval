import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from rag_retrieval_eval import main


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def write(self, name, data):
        path = self.root / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return str(path)

    def test_scores_recall_and_first_hit_at_cutoff(self):
        truth = self.write("truth.json", {"q1": ["a", "b"], "q2": ["c"]})
        run = self.write("run.json", {"q1": ["wrong", "a", "b"], "q2": []})
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            status = main(["score", truth, run, "--k", "2", "--min-recall", "0.5"])
        self.assertEqual(status, 1)
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["mean_recall"], 0.25)
        self.assertEqual(report["mean_reciprocal_rank"], 0.25)
        self.assertEqual(report["hit_rate"], 0.5)

    def test_rejects_missing_queries_and_duplicate_ranks(self):
        truth = self.write("truth.json", {"q1": ["a"], "q2": ["b"]})
        for invalid in ({"q1": ["a"]}, {"q1": ["a", "a"], "q2": []}):
            with self.subTest(invalid=invalid), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main(["score", truth, self.write("bad.json", invalid)])
                self.assertEqual(error.exception.code, 2)

    def test_ranks_portuguese_nfc_and_breaks_ties_by_id(self):
        corpus = self.write("docs.json", {"b": "ação", "a": "AÇÃO", "c": "controle"})
        queries = self.write("queries.json", {"q1": "ação", "q2": "nenhum"})
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(main(["rank", corpus, queries, "--k", "2"]), 0)
        self.assertEqual(json.loads(stdout.getvalue()), {"q1": ["a", "b"], "q2": []})

    def test_compare_baseline_on_the_same_judgments(self):
        truth = self.write("truth.json", {"q": ["a"]})
        new = self.write("new.json", {"q": ["a"]})
        old = self.write("old.json", {"q": ["b", "a"]})
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(main(["score", truth, new, "--baseline", old, "--k", "2"]), 0)
        self.assertEqual(json.loads(stdout.getvalue())["delta"]["mean_reciprocal_rank"], 0.5)

    def test_comparison_example_reports_expected_metrics_at_two_cutoffs(self):
        examples = Path(__file__).parents[1] / "examples"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(main([
                "rank", str(examples / "corpus.json"),
                str(examples / "queries-comparison.json"), "--k", "3"
            ]), 0)
        rankings = self.write("bm25.json", json.loads(stdout.getvalue()))

        for k, expected in (
            (2, (1.0, 1.0, 0.5, 0.5)),
            (3, (1.0, 1.0, 1.0, 0.5))
        ):
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(main([
                    "score", str(examples / "judgments-comparison.json"), rankings,
                    "--k", str(k), "--baseline", str(examples / "reference-ranking-comparison.json")
                ]), 0)
            report = json.loads(stdout.getvalue())
            self.assertEqual(
                (report["mean_recall"], report["mean_reciprocal_rank"],
                 report["baseline"]["mean_recall"], report["baseline"]["mean_reciprocal_rank"]),
                expected
            )
