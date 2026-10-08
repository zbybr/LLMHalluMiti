import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from common import config


class RootConfigurationTests(unittest.TestCase):
    def test_default_path_is_the_repository_root(self):
        self.assertEqual(config.ROOT_ENV, Path(__file__).resolve().parents[1] / ".env")

    def test_file_values_override_stale_process_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            env = Path(directory) / ".env"
            env.write_text(
                '\ufeffOPENAI_API_KEY="test-file-key"\n'
                'OPENAI_BASE_URL=https\\://example.invalid/v1\n'
                'export OLLAMA_BASE_URL="http://example.invalid:11434"\n',
                encoding="utf-8",
            )
            with mock.patch.object(config, "ROOT_ENV", env), mock.patch.dict(
                os.environ, {"OPENAI_API_KEY": "stale-process-key"}, clear=True
            ):
                self.assertTrue(config.load_root_env())
                self.assertEqual(os.environ["OPENAI_API_KEY"], "test-file-key")
                self.assertEqual(os.environ["OPENAI_BASE_URL"], "https://example.invalid/v1")
                self.assertEqual(os.environ["OLLAMA_BASE_URL"], "http://example.invalid:11434")

    def test_loader_does_not_search_the_current_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root_env = Path(directory) / "root.env"
            root_env.write_text("OPENAI_API_KEY=root-value\n", encoding="utf-8")
            with mock.patch.object(config, "ROOT_ENV", root_env), mock.patch.dict(
                os.environ, {}, clear=True
            ), mock.patch("os.getcwd", return_value="unrelated-directory"):
                config.load_root_env()
                self.assertEqual(os.environ["OPENAI_API_KEY"], "root-value")

    def test_simple_parser_preserves_quoted_hashes_and_strips_comments(self):
        with tempfile.TemporaryDirectory() as directory:
            env = Path(directory) / ".env"
            env.write_text('A="contains#hash"\nB=value # comment\nC=${B}/suffix\n', encoding="utf-8")
            self.assertEqual(
                config._read_simple_env(env),
                {"A": "contains#hash", "B": "value", "C": "value/suffix"},
            )

    def test_missing_file_fails_without_disclosing_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                config.read_env(Path(directory) / "missing.env")


if __name__ == "__main__":
    unittest.main()
