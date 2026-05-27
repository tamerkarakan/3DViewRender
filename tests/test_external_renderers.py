import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from external_renderers import resolve_blender_executable


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


if __name__ == "__main__":
    unittest.main()
