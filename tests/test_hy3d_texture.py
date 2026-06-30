import argparse
import importlib.util
import json
import pathlib
import sys
import tempfile
import types
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "hy3d_texture.py"


def load_texture_module():
    spec = importlib.util.spec_from_file_location("hy3d_texture_test_module", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TextureScriptTests(unittest.TestCase):
    def test_export_error_preserves_existing_output_and_records_error(self):
        module = load_texture_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            mesh_path = root / "input.glb"
            image_path = root / "input.png"
            output_path = root / "output.glb"
            source_root = root / "source"
            model_path = root / "models"
            metadata_path = root / "output.json"
            log_path = root / "output.log"

            mesh_path.write_bytes(b"mesh")
            image_path.write_bytes(b"image")
            output_path.write_bytes(b"existing output")
            (model_path / "hunyuan3d-paintpbr-v2-1").mkdir(parents=True)
            checkpoint = source_root / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")

            args = argparse.Namespace(
                mesh=str(mesh_path),
                image=str(image_path),
                out=str(output_path),
                source_root=str(source_root),
                model_path=str(model_path),
                device="cpu",
                max_views=6,
                resolution=512,
                no_remesh=True,
                dry_run=False,
                log=str(log_path),
                metadata=str(metadata_path),
            )
            module.parse_args = lambda: args
            module.resolve_paths = lambda _: (source_root, model_path)
            module.add_source_paths = lambda _: None
            module.patch_snapshot_download = lambda _: None
            module.install_bpy_stub_if_needed = lambda _: None
            module.add_windows_dll_dirs = lambda _: None

            torch_module = types.ModuleType("torch")
            torch_module.__version__ = "test"
            torch_module.cuda = types.SimpleNamespace(
                is_available=lambda: False,
                get_device_name=lambda _: "none",
            )

            trimesh_module = types.ModuleType("trimesh")

            class FailingScene:
                @staticmethod
                def export(path):
                    pathlib.Path(path).write_bytes(b"partial output")
                    raise RuntimeError("texture export exploded")

            trimesh_module.load = lambda *args, **kwargs: FailingScene()

            pipeline_module = types.ModuleType("textureGenPipeline")

            class FakeConfig:
                def __init__(self, max_views, resolution):
                    self.max_views = max_views
                    self.resolution = resolution

            class FakePipeline:
                def __init__(self, config):
                    self.config = config

                def __call__(self, **kwargs):
                    produced = pathlib.Path(kwargs["output_mesh_path"])
                    produced.write_bytes(b"obj")
                    return str(produced)

            pipeline_module.Hunyuan3DPaintConfig = FakeConfig
            pipeline_module.Hunyuan3DPaintPipeline = FakePipeline

            replacements = {
                "torch": torch_module,
                "trimesh": trimesh_module,
                "textureGenPipeline": pipeline_module,
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
            self.assertNotIn("output_size", metadata)
            self.assertIn("texture export exploded", metadata["error"])
            self.assertEqual(output_path.read_bytes(), b"existing output")
            self.assertFalse(pathlib.Path(str(output_path) + ".partial").exists())


if __name__ == "__main__":
    unittest.main()
