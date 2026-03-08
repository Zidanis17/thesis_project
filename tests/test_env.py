import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from thesis._env import load_project_env


class EnvLoaderTests(unittest.TestCase):
    def test_load_project_env_reads_dotenv_file_without_overwriting_existing_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "OPENAI_API_KEY=from-dotenv\nOTHER_VALUE=loaded\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"OPENAI_API_KEY": "from-process"}, clear=True):
                load_project_env(env_path)
                self.assertEqual(os.environ["OPENAI_API_KEY"], "from-process")
                self.assertEqual(os.environ["OTHER_VALUE"], "loaded")

    def test_load_project_env_can_override_existing_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text("OPENAI_API_KEY=from-dotenv\n", encoding="utf-8")

            with patch.dict(os.environ, {"OPENAI_API_KEY": "from-process"}, clear=True):
                load_project_env(env_path, override=True)
                self.assertEqual(os.environ["OPENAI_API_KEY"], "from-dotenv")


if __name__ == "__main__":
    unittest.main()
