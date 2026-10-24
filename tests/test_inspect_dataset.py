import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "src" / "inspect_dataset.py"


class DatasetLocationTests(unittest.TestCase):
    def test_downloads_beside_script_and_reuses_existing_dataset(self):
        sample = [{
            "sample_id": "test",
            "qa": [{"question": "Example?", "answer": "Yes"}],
            "conversation": {"session_1": [{"text": "Hello"}]},
        }]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            script = root / "src" / "inspect_dataset.py"
            script.parent.mkdir()
            script.write_text(SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
            expected_path = script.parent / "data" / "locomo10.json"
            working_directory = root / "elsewhere"
            working_directory.mkdir()

            def download(url, filename):
                self.assertEqual(Path(filename), expected_path)
                self.assertTrue(expected_path.parent.is_dir())
                expected_path.write_text(json.dumps(sample), encoding="utf-8")
                return str(expected_path), None

            previous_directory = Path.cwd()
            try:
                os.chdir(working_directory)
                with patch("urllib.request.urlretrieve", side_effect=download) as retrieve:
                    with contextlib.redirect_stdout(io.StringIO()) as output:
                        runpy.run_path(str(script), run_name="__main__")
                        runpy.run_path(str(script), run_name="__main__")
                    retrieve.assert_called_once()
                self.assertIn("Total conversations: 1", output.getvalue())
                self.assertFalse((working_directory / "data").exists())
            finally:
                os.chdir(previous_directory)


if __name__ == "__main__":
    unittest.main()
