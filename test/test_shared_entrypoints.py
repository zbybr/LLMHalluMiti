import contextlib
import io
import os
import runpy
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from common.launcher import launch
from common.models import MODEL_SPECS
from common import prompts
from common.stability import DEFAULT_DATASETS, extract_mutations


ROOT = Path(__file__).resolve().parents[1]


def dependency_stubs():
    pandas = types.ModuleType("pandas")
    pandas.DataFrame = type("DataFrame", (), {})
    pandas.Series = type("Series", (), {})
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *args, **kwargs: None
    openai = types.ModuleType("openai")
    openai.OpenAI = lambda *args, **kwargs: object()
    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda iterable=None, *args, **kwargs: iterable
    return {
        "pandas": pandas,
        "dotenv": dotenv,
        "openai": openai,
        "tqdm": tqdm_module,
    }


class ModelConfigurationTests(unittest.TestCase):
    def test_all_four_models_have_one_shared_spec(self):
        self.assertEqual(set(MODEL_SPECS), {"gpt-4o", "gpt-5", "gemini", "qwen3"})
        self.assertEqual(MODEL_SPECS["gemini"].output_prefix, "gemini")
        self.assertEqual(
            MODEL_SPECS["qwen3"].extra_body, {"enable_thinking": False}
        )

    def test_launcher_derives_model_from_legacy_path(self):
        with mock.patch("runpy.run_module") as run_module:
            launch("leetcode.cot", str(ROOT / "gemini" / "cot_leetcode.py"))
        run_module.assert_called_once_with("leetcode.cot", run_name="__main__")
        self.assertEqual(os.environ["LLM_MODEL_KEY"], "gemini-2.5-flash-thinking")
        self.assertEqual(os.environ["LLM_OUTPUT_PREFIX"], "gemini")
        self.assertEqual(os.environ["LLM_ENV_FILE"], str(ROOT / ".env"))

    def test_all_models_use_the_same_structured_qa_prompt(self):
        expected = prompts.MUTATION_PROMPT.format(n=5)
        self.assertEqual(prompts.qa_mutation_prompt(), expected)
        self.assertNotIn("{n}", prompts.qa_mutation_prompt())
        for spec in MODEL_SPECS.values():
            self.assertFalse(hasattr(spec, "qa_mutation_prompt"))

    def test_refine_prompt_is_extraction_only(self):
        self.assertIn("strict answer extractor", prompts.REFINE_PROMPT)
        self.assertIn("Do not\nverify, correct, re-rank", prompts.REFINE_PROMPT)
        self.assertIn("ORIGINAL CANDIDATES", prompts.REFINE_PROMPT)
        self.assertIn("Return exactly one complete answer sentence", prompts.REFINE_PROMPT)

    def test_stability_uses_full_datasets_and_exactly_five_mutations(self):
        self.assertEqual(set(DEFAULT_DATASETS), set(MODEL_SPECS))
        self.assertTrue(all("sampled" not in name for name in DEFAULT_DATASETS.values()))
        mutations = extract_mutations("1. one\n2. two\n3. three\n4. four\n5. five")
        self.assertEqual(mutations, ["one", "two", "three", "four", "five"])


class LegacyEntrypointTests(unittest.TestCase):
    def test_all_legacy_mainline_commands_still_offer_help(self):
        scripts = []
        for model in ("gpt-4o", "gpt-5", "gemini", "qwen3"):
            model_dir = ROOT / model
            scripts.extend(
                [
                    model_dir / "cot_leetcode.py",
                    model_dir / "drhall_leetcode.py",
                    model_dir / "repair_with_mutation_leetcode.py",
                    model_dir / "auto_evaluation_leetcode.py",
                    model_dir / "tools" / "generate_response_leetcode.py",
                    model_dir / "tools" / "generate_response.py",
                    model_dir / "cot.py",
                    model_dir / "repair.py",
                    model_dir / "drhall.py",
                    model_dir / "repair_with_mutation.py",
                    model_dir / "repair_with_mutation_stability.py",
                ]
            )

        old_argv = sys.argv[:]
        old_path = sys.path[:]
        try:
            with mock.patch.dict(sys.modules, dependency_stubs()):
                for script in scripts:
                    with self.subTest(script=str(script.relative_to(ROOT))):
                        sys.argv = [str(script), "--help"]
                        sys.path.insert(0, str(script.parent))
                        with contextlib.redirect_stdout(io.StringIO()):
                            with self.assertRaises(SystemExit) as exit_result:
                                runpy.run_path(str(script), run_name="__main__")
                        self.assertEqual(exit_result.exception.code, 0)
        finally:
            sys.argv = old_argv
            sys.path[:] = old_path


if __name__ == "__main__":
    unittest.main()
