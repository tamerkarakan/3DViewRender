import unittest

import nodes
from renderer import VIEW_ORDER


class UiContractTests(unittest.TestCase):
    def test_node_outputs_image_batch_and_view_names(self):
        self.assertEqual(nodes.SixSideRender.RETURN_TYPES, ("IMAGE", "STRING"))
        self.assertEqual(nodes.SixSideRender.RETURN_NAMES, ("images", "view_names"))

    def test_legacy_ui_schema_exposes_six_side_toggles(self):
        schema = nodes.SixSideRender.INPUT_TYPES()
        required = schema["required"]

        for side in VIEW_ORDER:
            self.assertIn(side, required)
            self.assertEqual(required[side][0], "BOOLEAN")
            self.assertIs(required[side][1]["default"], True)

    def test_legacy_ui_schema_exposes_camera_modes_and_render_controls(self):
        required = nodes.SixSideRender.INPUT_TYPES()["required"]

        self.assertEqual(required["mesh"], ("MESH",))
        self.assertEqual(required["camera_mode"][0], ["orthographic", "perspective"])
        self.assertEqual(required["camera_mode"][1]["default"], "orthographic")
        self.assertEqual(required["resolution"][1]["default"], 512)
        self.assertIn("fov_degrees", required)
        self.assertIn("orthographic_scale", required)
        self.assertIn("camera_distance", required)


if __name__ == "__main__":
    unittest.main()
