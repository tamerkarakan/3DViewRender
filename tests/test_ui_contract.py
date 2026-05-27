import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nodes
from renderer import VIEW_ORDER


class UiContractTests(unittest.TestCase):
    def test_node_outputs_image_batch_and_view_names(self):
        self.assertEqual(tuple(nodes.SixSideRender.RETURN_TYPES), ("IMAGE", "STRING", "STRING"))
        self.assertEqual(tuple(nodes.SixSideRender.RETURN_NAMES), ("images", "view_names", "render_info"))

    def test_legacy_ui_schema_exposes_six_side_toggles(self):
        schema = nodes._legacy_inputs()
        required = schema["required"]

        for side in VIEW_ORDER:
            self.assertIn(side, required)
            self.assertEqual(required[side][0], "BOOLEAN")
            self.assertIs(required[side][1]["default"], True)

    def test_legacy_ui_schema_exposes_camera_modes_and_render_controls(self):
        required = nodes._legacy_inputs()["required"]

        self.assertIn("MESH", required["model"][0])
        self.assertIn("TRIMESH", required["model"][0])
        self.assertIn("MESHWITHVOXEL", required["model"][0])
        self.assertIn("FILE_3D_GLB", required["model"][0])
        self.assertIn("STRING", required["model"][0])
        self.assertEqual(
            required["renderer_backend"][0],
            ["cpu_preview", "f3d_optional", "blender_optional", "nvdiffrast_optional"],
        )
        self.assertEqual(required["renderer_backend"][1]["default"], "cpu_preview")
        self.assertEqual(required["auto_install_f3d"][1]["default"], False)
        self.assertTrue(required["auto_install_f3d"][1]["advanced"])
        self.assertEqual(required["blender_path"][1]["default"], "")
        self.assertTrue(required["blender_path"][1]["advanced"])
        self.assertEqual(required["camera_mode"][0], ["orthographic", "perspective"])
        self.assertEqual(required["camera_mode"][1]["default"], "orthographic")
        self.assertEqual(required["up_axis"][0], ["z_up", "y_up"])
        self.assertEqual(required["up_axis"][1]["default"], "z_up")
        self.assertEqual(required["resolution"][1]["default"], 512)
        self.assertEqual(required["max_faces"][1]["default"], 10000)
        self.assertTrue(required["max_faces"][1]["advanced"])
        self.assertIn("fov_degrees", required)
        self.assertIn("orthographic_scale", required)
        self.assertIn("camera_distance", required)

    def test_render_info_names_backend_engine_and_orientation(self):
        info = nodes._format_render_info(
            renderer_backend="cpu_preview",
            settings=nodes.RenderSettings(width=128, height=128, up_axis="z_up"),
            views=["right"],
            max_faces=10000,
        )

        self.assertIn("renderer_backend=cpu_preview", info)
        self.assertIn("renderer_engine=MeshRenderer CPU rasterizer", info)
        self.assertIn("up_axis=z_up", info)
        self.assertIn("views=right", info)

    def test_render_info_ui_is_frontend_text_payload(self):
        self.assertEqual(
            nodes._render_info_ui("renderer_backend=cpu_preview\nup_axis=z_up"),
            {"text": ["renderer_backend=cpu_preview", "up_axis=z_up"]},
        )


if __name__ == "__main__":
    unittest.main()
