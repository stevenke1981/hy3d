import argparse
import contextlib
import importlib.util
import json
import pathlib
import sys
import tempfile
import types
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "hy3d_generate.py"


def load_generate_module():
    spec = importlib.util.spec_from_file_location("hy3d_generate_test_module", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class GenerateScriptTests(unittest.TestCase):
    def test_preflight_reports_missing_image_without_importing_dependencies(self):
        module = load_generate_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            args = argparse.Namespace(subfolder="shape")
            result = module.validate_preflight(
                args,
                root / "missing.png",
                root / "source",
                root / "models",
            )
            self.assertEqual(result[0], 2)
            self.assertIn("image not found", result[1])

    def test_dependency_import_failure_is_recorded(self):
        module = load_generate_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            image_path = root / "input.png"
            output_path = root / "output.glb"
            source_root = root / "source"
            model_path = root / "models"
            metadata_path = root / "output.json"
            image_path.write_bytes(b"image")
            source_root.mkdir()
            checkpoint = model_path / "shape" / "model.fp16.ckpt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            args = argparse.Namespace(
                image=str(image_path), out=str(output_path), source_root=str(source_root),
                model_path=str(model_path), subfolder="shape", device="cpu", quality="smoke",
                steps=1, seed=42, low_vram=False, no_rembg=True, dry_run=False,
                log=str(root / "output.log"), metadata=str(metadata_path),
            )
            module.parse_args = lambda: args
            module.resolve_paths = lambda _: (source_root, model_path)
            module.add_source_paths = lambda _: None

            def fail_import():
                raise RuntimeError("import exploded")

            module.load_dependencies = fail_import
            result = module.main()
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(result, 5)
            self.assertEqual(metadata["status"], "error")
            self.assertIn("import exploded", metadata["error"])

    def test_unhandled_inference_error_is_recorded_in_sidecar(self):
        module = load_generate_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            image_path = root / "input.png"
            output_path = root / "output.glb"
            source_root = root / "source"
            model_path = root / "models"
            metadata_path = root / "output.json"
            log_path = root / "output.log"

            image_path.write_bytes(b"image")
            output_path.write_bytes(b"existing output")
            source_root.mkdir()
            checkpoint = model_path / "shape" / "model.fp16.ckpt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")

            args = argparse.Namespace(
                image=str(image_path),
                out=str(output_path),
                source_root=str(source_root),
                model_path=str(model_path),
                subfolder="shape",
                device="cpu",
                quality="smoke",
                steps=1,
                seed=42,
                low_vram=False,
                no_rembg=True,
                dry_run=False,
                log=str(log_path),
                metadata=str(metadata_path),
            )
            module.parse_args = lambda: args
            module.resolve_paths = lambda _: (source_root, model_path)
            module.add_source_paths = lambda _: None

            torch_module = types.ModuleType("torch")
            torch_module.__version__ = "test"
            torch_module.float16 = "float16"
            torch_module.float32 = "float32"
            torch_module.manual_seed = lambda _: None
            torch_module.inference_mode = contextlib.nullcontext
            torch_module.cuda = types.SimpleNamespace(
                is_available=lambda: False,
                get_device_name=lambda _: "none",
                manual_seed_all=lambda _: None,
            )

            pil_module = types.ModuleType("PIL")
            pil_image_module = types.ModuleType("PIL.Image")
            pil_module.Image = pil_image_module

            pipelines_module = types.ModuleType("hy3dshape.pipelines")

            class FailingPipeline:
                @classmethod
                def from_pretrained(cls, *args, **kwargs):
                    return cls()

                def __call__(self, *args, **kwargs):
                    class FailingMesh:
                        @staticmethod
                        def export(path):
                            pathlib.Path(path).write_bytes(b"partial output")
                            raise RuntimeError("export exploded")

                    return [FailingMesh()]

            pipelines_module.Hunyuan3DDiTFlowMatchingPipeline = FailingPipeline
            hy3dshape_module = types.ModuleType("hy3dshape")
            hy3dshape_module.pipelines = pipelines_module

            replacements = {
                "torch": torch_module,
                "PIL": pil_module,
                "PIL.Image": pil_image_module,
                "hy3dshape": hy3dshape_module,
                "hy3dshape.pipelines": pipelines_module,
            }
            previous = {name: sys.modules.get(name) for name in replacements}
            sys.modules.update(replacements)
            try:
                result = module.main()
            finally:
                for name, value in previous.items():
                    if value is None:
                        sys.modules.pop(name, None)
                    else:
                        sys.modules[name] = value

            self.assertEqual(result, 99)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "error")
            self.assertEqual(metadata["exit_code"], 99)
            self.assertIn("export exploded", metadata["error"])
            self.assertNotIn("output_size", metadata)
            self.assertEqual(output_path.read_bytes(), b"existing output")
            self.assertFalse(pathlib.Path(str(output_path) + ".partial").exists())


if __name__ == "__main__":
    unittest.main()
