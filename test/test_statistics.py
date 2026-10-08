"""Offline regression tests for the public, read-only statistics scripts."""

import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

STAT = Path(__file__).resolve().parents[1] / "stat"
sys.path.insert(0, str(STAT))
from metrics import metric_counts, metrics_for_file, parse_success, write_table
from mcnemar_test import compare_paper_metrics, exact_mcnemar
from run_all_mcnemar import holm_adjust, normalized_question, validate_row_pairing


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class StatisticsTests(unittest.TestCase):
    def test_empty_is_incorrect_under_both_column_semantics(self):
        for semantics in ("correct", "incorrect"):
            for value in (None, "", "  ", "\t"):
                self.assertEqual(parse_success(value, semantics), 0)
        self.assertEqual(parse_success("YES", "incorrect"), 0)
        self.assertEqual(parse_success("NO", "incorrect"), 1)
        self.assertEqual(parse_success("NN", "incorrect"), 1)
        self.assertEqual(parse_success("pass", "correct"), 1)
        self.assertEqual(parse_success("fail", "correct"), 0)
        with self.assertRaises(ValueError):
            parse_success("unreviewed", "incorrect")

    def test_metric_formulas_and_zero_denominators(self):
        result = metric_counts([0, 0, 1, 1], [1, 0, 0, 1])
        for name in ("HRR", "RHR", "OCR"):
            self.assertEqual(result[name], 50)
        self.assertIsNone(metric_counts([1], [1])["HRR"])
        self.assertIsNone(metric_counts([0], [1])["OCR"])

    def test_exact_p_value_and_holm(self):
        self.assertEqual(exact_mcnemar([1] * 7, [0] * 7)["exact_mcnemar_p_two_sided"], 0.015625)
        self.assertEqual(exact_mcnemar([1, 0], [1, 0])["exact_mcnemar_p_two_sided"], 1)
        self.assertEqual(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_empty_labels_count_as_failure_in_all_three_paired_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a.csv", Path(directory) / "b.csv"
            fields = ["Question", "is_hallucination", "recheck_hallucination"]
            write_csv(a, fields, [dict(zip(fields, values)) for values in
                                 [("q1", "", ""), ("q2", "NO", "")]])
            write_csv(b, fields, [dict(zip(fields, values)) for values in
                                 [("q1", "", "NO"), ("q2", "NO", "NO")]])
            before = (a.read_bytes(), b.read_bytes())
            report = compare_paper_metrics(a, b, bootstrap_iterations=20)
            self.assertEqual(report["metrics"]["HRR"]["rate_a"], 0)
            self.assertEqual(report["metrics"]["HRR"]["rate_b"], 1)
            self.assertEqual(report["metrics"]["RHR"]["rate_a"], 1)
            self.assertEqual(report["metrics"]["OCR"]["rate_a"], 1)
            self.assertEqual(report["metrics"]["RHR"]["n_pairs"], 2)
            self.assertEqual(before, (a.read_bytes(), b.read_bytes()))

    def test_pass_schema_and_embedded_summary_are_recomputed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stability.csv"
            write_csv(path, ["id", "base", "mutrepair", "HRR"],
                      [{"id": "1", "base": "", "mutrepair": "pass", "HRR": "999"},
                       {"id": "2", "base": "pass", "mutrepair": "", "HRR": ""}])
            result = metrics_for_file(path)
            self.assertEqual(result["HRR"], 100)
            self.assertEqual(result["OCR"], 100)
            self.assertEqual(result["RHR"], 50)
            write_csv(path, ["task_id", "base_pass", "mutrepair_pass"],
                      [{"task_id": "x", "base_pass": "fail", "mutrepair_pass": ""}])
            self.assertEqual(metrics_for_file(path)["RHR"], 100)

    def test_mismatched_base_and_duplicate_ids_are_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a.csv", Path(directory) / "b.csv"
            fields = ["id", "base", "mutrepair"]
            write_csv(a, fields, [{"id": "1", "base": "pass", "mutrepair": "pass"}])
            write_csv(b, fields, [{"id": "1", "base": "", "mutrepair": "pass"}])
            with self.assertRaisesRegex(ValueError, "base outcome differs"):
                compare_paper_metrics(a, b, bootstrap_iterations=20)
            write_csv(a, fields, [{"id": "1", "base": "pass", "mutrepair": "pass"}] * 2)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                metrics_for_file(a)

    def test_question_validation_handles_legacy_accents_but_rejects_reordering(self):
        self.assertEqual(normalized_question("Pok\u00c3\u0083\u00c2\u00a9mon"), "Pokemon")
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a.csv", Path(directory) / "b.csv"
            write_csv(a, ["Question"], [{"Question": "a"}, {"Question": "b"}])
            write_csv(b, ["Question"], [{"Question": "b"}, {"Question": "a"}])
            with self.assertRaisesRegex(ValueError, "validation"):
                validate_row_pairing(a, b)

    def test_summary_cannot_overwrite_model_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "outputs" / "source.csv"
            with self.assertRaisesRegex(ValueError, "overwrite"):
                write_table(path, [{"x": 1}])

    def test_cli_help_requires_no_credentials_or_model_dependencies(self):
        for name in ("mcnemar_test.py", "run_all_mcnemar.py", "summarize_results.py",
                     "finalize_stability_metrics.py"):
            result = subprocess.run([sys.executable, "-B", str(STAT / name), "--help"],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
