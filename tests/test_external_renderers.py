import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from external_renderers import _write_blender_config, resolve_blender_executable
from renderer import CameraMode, RenderSettings


class ExternalRendererTests(unittest.TestCase):
    def test_resolve_blender_executable_accepts_folder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            blender_exe = Path(temp_dir) / "blender.exe"
            blender_exe.write_text("", encoding="ascii")

            self.assertEqual(resolve_blender_executable(temp_dir), str(blender_exe))

    def test_resolve_blender_executable_accepts_exe_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            blender_exe = Path(temp_dir) / "blender.exe"
            blender_exe.write_text("", encoding="ascii")

            self.assertEqual(resolve_blender_executable(str(blender_exe)), str(blender_exe))

    def test_blender_config_uses_resolved_front_axis_view_poses(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "render_config.json"
            output_dir = Path(temp_dir) / "images"
            settings = RenderSettings(
                width=64,
                height=64,
                camera_mode=CameraMode.ORTHOGRAPHIC,
                up_axis="+Y",
                front_axis="+Z",
            )

            _write_blender_config(config_path, "model.glb", ["front", "right"], settings, output_dir)

            config = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(config["front_axis"], "+Z")
            self.assertEqual(config["view_poses"]["front"]["direction"], [0.0, 0.0, 1.0])
            self.assertEqual(config["view_poses"]["right"]["direction"], [1.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
