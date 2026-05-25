import importlib.util
import pathlib
import struct
import sys
import tempfile
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "convert-hy3d-shape-to-gguf.py"


def load_converter():
    spec = importlib.util.spec_from_file_location("hy3d_converter", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ConverterTests(unittest.TestCase):
    def test_writer_creates_valid_gguf_with_metadata_and_tensor(self):
        converter = load_converter()
        with tempfile.TemporaryDirectory() as tmp:
            out_path = pathlib.Path(tmp) / "tiny.gguf"
            tensors = [
                converter.TensorRecord(
                    name="blocks.0.weight",
                    array=np.arange(6, dtype=np.float32).reshape(2, 3),
                )
            ]
            metadata = {
                "general.architecture": "hunyuan3d",
                "hy3d.model_part": "shape",
                "hy3d.tensor_layout": "pytorch",
            }

            converter.write_gguf(out_path, tensors, metadata, alignment=32)

            data = out_path.read_bytes()
            self.assertEqual(data[:4], b"GGUF")
            version, tensor_count, metadata_count = struct.unpack_from("<IQQ", data, 4)
            self.assertEqual(version, 3)
            self.assertEqual(tensor_count, 1)
            self.assertEqual(metadata_count, 3)

    def test_map_tensor_name_strips_common_checkpoint_prefixes(self):
        converter = load_converter()
        self.assertEqual(converter.map_tensor_name("state_dict.model.blocks.0.weight"), "blocks.0.weight")
        self.assertEqual(converter.map_tensor_name("module.model.final_layer.bias"), "final_layer.bias")


if __name__ == "__main__":
    unittest.main()
