import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "write_dependency_manifest.py"


class DependencyManifestTests(unittest.TestCase):
    def test_writes_atomic_sorted_environment_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = pathlib.Path(tmp) / "manifest.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--out",
                    str(output),
                    "--source-revision",
                    "source-test",
                    "--model-revision",
                    "model-test",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(manifest["source_revision"], "source-test")
            self.assertEqual(manifest["model_revision"], "model-test")
            self.assertEqual(manifest["packages"], dict(sorted(manifest["packages"].items())))
            self.assertIn("python", manifest)
            self.assertFalse(pathlib.Path(str(output) + ".partial").exists())


if __name__ == "__main__":
    unittest.main()
