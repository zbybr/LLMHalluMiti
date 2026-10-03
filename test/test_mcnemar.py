import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "stat" / "mcnemar_test.py"
SPEC = importlib.util.spec_from_file_location("mutrepair_mcnemar", MODULE_PATH)
MCNEMAR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MCNEMAR)


class McNemarTests(unittest.TestCase):
    def test_exact_mcnemar_uses_only_discordant_pairs(self):
        result = MCNEMAR.exact_mcnemar(
            [0, 0, 1, 1, 1, 0],
            [1, 1, 0, 1, 1, 0],
        )
        self.assertEqual(result["a_success_b_failure"], 1)
        self.assertEqual(result["a_failure_b_success"], 2)
        self.assertEqual(result["discordant_pairs"], 3)
        self.assertEqual(result["exact_mcnemar_p_two_sided"], 1.0)

    def test_incorrect_semantics_converts_yes_to_failure(self):
        self.assertEqual(MCNEMAR.parse_success("YES", "incorrect"), 0)
        self.assertEqual(MCNEMAR.parse_success("NO", "incorrect"), 1)

    def test_bootstrap_is_reproducible(self):
        first = MCNEMAR.paired_bootstrap_difference(
            [0, 0, 1, 1], [1, 1, 0, 1], iterations=100, seed=7
        )
        second = MCNEMAR.paired_bootstrap_difference(
            [0, 0, 1, 1], [1, 1, 0, 1], iterations=100, seed=7
        )
        self.assertEqual(first, second)

    def test_paper_mode_auto_detects_qa_and_reports_all_metrics(self):
        headers = ["Question", "is_hallucination", "recheck_hallucination"]
        rows_a = [
            ["q1", "YES", "YES"],
            ["q2", "YES", "NO"],
            ["q3", "NO", "NO"],
            ["q4", "NO", "YES"],
        ]
        rows_b = [
            ["q1", "YES", "NO"],
            ["q2", "YES", "NO"],
            ["q3", "NO", "YES"],
            ["q4", "NO", "YES"],
        ]
        with tempfile.TemporaryDirectory() as directory:
            file_a = Path(directory) / "a.csv"
            file_b = Path(directory) / "b.csv"
            for path, rows in ((file_a, rows_a), (file_b, rows_b)):
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(headers)
                    writer.writerows(rows)

            report = MCNEMAR.compare_paper_metrics(
                file_a, file_b, bootstrap_iterations=100, seed=7
            )

        self.assertEqual(report["schema"], "qa")
        self.assertEqual(report["metrics"]["HRR"]["n_pairs"], 2)
        self.assertEqual(report["metrics"]["HRR"]["rate_a"], 0.5)
        self.assertEqual(report["metrics"]["HRR"]["rate_b"], 1.0)
        self.assertEqual(report["metrics"]["OCR"]["n_pairs"], 2)
        self.assertEqual(report["metrics"]["OCR"]["rate_a"], 0.5)
        self.assertEqual(report["metrics"]["OCR"]["rate_b"], 1.0)

    def test_paper_mode_auto_detects_code(self):
        with tempfile.TemporaryDirectory() as directory:
            file_a = Path(directory) / "a.csv"
            file_b = Path(directory) / "b.csv"
            file_a.write_text(
                "task_id,base_pass,a_final_pass\nt1,False,False\nt2,True,True\n",
                encoding="utf-8",
            )
            file_b.write_text(
                "task_id,base_pass,b_final_pass\nt1,False,True\nt2,True,True\n",
                encoding="utf-8",
            )
            report = MCNEMAR.compare_paper_metrics(
                file_a, file_b, bootstrap_iterations=100, seed=7
            )

        self.assertEqual(report["schema"], "code")
        self.assertEqual(report["metrics"]["HRR"]["rate_a"], 0.0)
        self.assertEqual(report["metrics"]["HRR"]["rate_b"], 1.0)
        self.assertEqual(report["metrics"]["OCR"]["rate_a"], 0.0)


if __name__ == "__main__":
    unittest.main()
