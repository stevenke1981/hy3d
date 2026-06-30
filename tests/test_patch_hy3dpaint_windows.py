import pathlib
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PATCHER = REPO_ROOT / "scripts" / "patch_hy3dpaint_windows.py"


class WindowsRasterizerPatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.source = pathlib.Path(self.temp.name)
        kernel = self.source / "hy3dpaint/custom_rasterizer/lib/custom_rasterizer_kernel"
        kernel.mkdir(parents=True)
        (kernel / "grid_neighbor.cpp").write_text(
            """
torch::zeros({seq2pos.size() / 3, 3}, float_options);
torch::zeros({seq2feat.size() / feat_channel, feat_channel}, float_options);
torch::zeros({grids[i].seq2grid.size(), 9}, int64_options);
long* nptr = value.data_ptr<long>();
long* dptr = value.data_ptr<long>();
""",
            encoding="utf-8",
        )
        rasterizer = """
auto z_min = torch::ones({height, width}, INT64_options) * (long)maxint;
call((INT64*)z_min.data_ptr<long>());
"""
        (kernel / "rasterizer.cpp").write_text(rasterizer, encoding="utf-8")
        (kernel / "rasterizer_gpu.cu").write_text(rasterizer, encoding="utf-8")
        (self.source / "hy3dpaint/custom_rasterizer/setup.py").write_text(
            """
custom_rasterizer_module = CUDAExtension(
    sources=[
        "lib/custom_rasterizer_kernel/rasterizer_gpu.cu",
    ],
)
""",
            encoding="utf-8",
        )
        subprocess.run(["git", "init", "--quiet"], cwd=self.source, check=True)
        subprocess.run(["git", "add", "."], cwd=self.source, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=hy3d-test",
                "-c",
                "user.email=hy3d@example.invalid",
                "commit",
                "--quiet",
                "-m",
                "fixture",
            ],
            cwd=self.source,
            check=True,
        )
        self.revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=self.source, text=True
        ).strip()

    def tearDown(self):
        self.temp.cleanup()

    def run_patcher(self, revision):
        return subprocess.run(
            [
                sys.executable,
                str(PATCHER),
                "--source-root",
                str(self.source),
                "--expected-revision",
                revision,
            ],
            text=True,
            capture_output=True,
        )

    def test_patch_is_revision_guarded_and_idempotent(self):
        rejected = self.run_patcher("0" * 40)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("revision", rejected.stderr.lower())

        first = self.run_patcher(self.revision)
        self.assertEqual(first.returncode, 0, first.stderr)
        grid = (
            self.source
            / "hy3dpaint/custom_rasterizer/lib/custom_rasterizer_kernel/grid_neighbor.cpp"
        ).read_text(encoding="utf-8")
        self.assertIn("static_cast<int64_t>(seq2pos.size() / 3)", grid)
        self.assertIn("static_cast<int64_t>(seq2feat.size() / feat_channel)", grid)
        self.assertIn("int64_t* nptr", grid)
        self.assertIn("data_ptr<int64_t>()", grid)
        setup = (self.source / "hy3dpaint/custom_rasterizer/setup.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("-allow-unsupported-compiler", setup)

        snapshot = subprocess.check_output(
            ["git", "diff", "--"], cwd=self.source, text=True
        )
        second = self.run_patcher(self.revision)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(
            snapshot,
            subprocess.check_output(["git", "diff", "--"], cwd=self.source, text=True),
        )


if __name__ == "__main__":
    unittest.main()
