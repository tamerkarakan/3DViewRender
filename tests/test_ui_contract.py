import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nodes
from renderer import VIEW_ORDER


class UiContractTests(unittest.TestCase):
    def test_node_outputs_image_batch_and_view_names(self):
        self.assertEqual(tuple(nodes.SixSideRender.RETURN_TYPES), ("IMAGE", "STRING", "STRING", "IMAGE"))
        self.assertEqual(tuple(nodes.SixSideRender.RETURN_NAMES), ("images", "view_names", "render_info", "contact_sheet"))

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
        self.assertEqual(required["up_axis"][0], ["+Z", "-Z", "+Y", "-Y", "+X", "-X"])
        self.assertEqual(required["up_axis"][1]["default"], "+Z")
        self.assertEqual(required["matrix_layout"][0], ["3x2", "2x3", "6x1", "1x6", "auto"])
        self.assertEqual(required["matrix_layout"][1]["default"], "3x2")
        self.assertEqual(required["label_matrix"][1]["default"], True)
        self.assertEqual(required["save_to_output"][1]["default"], True)
        self.assertEqual(required["filename_prefix"][1]["default"], "3DViewRender/render")
        self.assertEqual(required["resolution"][1]["default"], 512)
        self.assertEqual(required["max_faces"][1]["default"], 10000)
        self.assertTrue(required["max_faces"][1]["advanced"])
        self.assertIn("fov_degrees", required)
        self.assertIn("orthographic_scale", required)
        self.assertIn("camera_distance", required)

    def test_render_info_names_backend_engine_and_orientation(self):
        info = nodes._format_render_info(
            renderer_backend="cpu_preview",
            settings=nodes.RenderSettings(width=128, height=128, up_axis="+Z"),
            views=["right"],
            max_faces=10000,
            matrix_layout="3x2",
            save_to_output=True,
            filename_prefix="3DViewRender/render",
        )

        self.assertIn("renderer_backend=cpu_preview", info)
        self.assertIn("renderer_engine=MeshRenderer CPU rasterizer", info)
        self.assertIn("up_axis=+Z", info)
        self.assertIn("views=right", info)
        self.assertIn("matrix_layout=3x2", info)
        self.assertIn("save_to_output=true", info)

    def test_single_batch_view_names_are_plain_side_names(self):
        self.assertEqual(nodes._view_output_name(0, 1, "front"), "front")
        self.assertEqual(nodes._view_output_name(2, 3, "front"), "mesh2_front")

    def test_build_settings_accepts_legacy_axis_aliases(self):
        self.assertEqual(
            nodes._build_settings(
                resolution=64,
                camera_mode="orthographic",
                background_color="#000000",
                mesh_color="#FFFFFF",
                up_axis="z_up",
                fov_degrees=45.0,
                orthographic_scale=1.2,
                camera_distance=2.4,
                shading=False,
            ).up_axis,
            "+Z",
        )

    def test_render_info_ui_is_frontend_text_payload(self):
        self.assertEqual(
            nodes._render_info_ui("renderer_backend=cpu_preview\nup_axis=+Z"),
            {"text": ["renderer_backend=cpu_preview", "up_axis=+Z"]},
        )


if __name__ == "__main__":
    unittest.main()
