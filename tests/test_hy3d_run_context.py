import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from hy3d_run_context import RunContext, sidecar_path  # noqa: E402


class RunContextTests(unittest.TestCase):
    def test_error_metadata_is_written_and_streams_are_restored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            output = root / "result.glb"
            log = root / "result.log"
            metadata_path = root / "result.json"
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            metadata = {"command": "test", "status": "started"}

            with RunContext(output, log, metadata_path, metadata) as run:
                print("captured output")
                code = run.finish("error", 7, "expected failure")

            self.assertEqual(code, 7)
            self.assertIs(sys.stdout, original_stdout)
            self.assertIs(sys.stderr, original_stderr)
            self.assertIn("captured output", log.read_text(encoding="utf-8"))
            saved = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "error")
            self.assertEqual(saved["exit_code"], 7)
            self.assertEqual(saved["error"], "expected failure")
            self.assertNotIn("output_size", saved)

    def test_success_records_output_size_and_sidecar_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = pathlib.Path(tmp) / "result.glb"
            output.write_bytes(b"mesh")
            metadata = {}
            with RunContext(
                output,
                sidecar_path(output, ".log.txt"),
                sidecar_path(output, ".json"),
                metadata,
            ) as run:
                self.assertEqual(run.finish("ok", 0), 0)
            self.assertEqual(metadata["output_size"], 4)


if __name__ == "__main__":
    unittest.main()
