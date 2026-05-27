import importlib.util
import unittest
import tempfile
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from renderer import (
    CameraMode,
    ContactSheetBuilder,
    ContactSheetSettings,
    MeshBatchItem,
    MeshRenderer,
    RenderSettings,
    parse_color,
    selected_view_names,
    view_pose,
)
from nodes import _file_type_from_model, _limit_faces, _mesh_batch_items


def make_box(width=1.0, height=0.6, depth=0.3):
    x = width / 2
    y = height / 2
    z = depth / 2
    vertices = np.array(
        [
            [-x, -y, -z],
            [x, -y, -z],
            [x, y, -z],
            [-x, y, -z],
            [-x, -y, z],
            [x, -y, z],
            [x, y, z],
            [-x, y, z],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [3, 2, 6],
            [3, 6, 7],
            [1, 5, 6],
            [1, 6, 2],
            [0, 3, 7],
            [0, 7, 4],
        ],
        dtype=np.int64,
    )
    return MeshBatchItem(vertices=vertices, faces=faces)


class MeshRendererTests(unittest.TestCase):
    def test_selected_views_follow_fixed_ui_order(self):
        self.assertEqual(
            selected_view_names(front=False, back=True, left=False, right=True, top=True, bottom=False),
            ["back", "right", "top"],
        )

    def test_parse_color_accepts_hex_and_rgb_triplets(self):
        self.assertEqual(parse_color("#FF8000", (0.0, 0.0, 0.0)), (1.0, 128 / 255.0, 0.0))
        self.assertEqual(parse_color("255, 0, 128", (0.0, 0.0, 0.0)), (1.0, 0.0, 128 / 255.0))
        self.assertEqual(parse_color("bad", (0.1, 0.2, 0.3)), (0.1, 0.2, 0.3))

    def test_orthographic_front_and_top_are_nonblank_and_distinct(self):
        renderer = MeshRenderer()
        settings = RenderSettings(
            width=96,
            height=96,
            camera_mode=CameraMode.ORTHOGRAPHIC,
            orthographic_scale=1.4,
            shading=False,
            mesh_color=(1.0, 1.0, 1.0),
        )
        renders = renderer.render_views(make_box(), ["front", "top"], settings)

        self.assertEqual(renders[0].image.shape, (96, 96, 3))
        self.assertGreater(float(renders[0].image.sum()), 100.0)
        self.assertGreater(float(renders[1].image.sum()), 100.0)
        self.assertFalse(np.allclose(renders[0].image, renders[1].image))

    def test_perspective_and_orthographic_projection_differ(self):
        renderer = MeshRenderer()
        mesh = make_box(width=1.0, height=0.8, depth=0.8)
        ortho = RenderSettings(
            width=96,
            height=96,
            camera_mode=CameraMode.ORTHOGRAPHIC,
            orthographic_scale=1.5,
            shading=False,
            mesh_color=(1.0, 1.0, 1.0),
        )
        perspective = RenderSettings(
            width=96,
            height=96,
            camera_mode=CameraMode.PERSPECTIVE,
            fov_degrees=45.0,
            camera_distance=2.0,
            shading=False,
            mesh_color=(1.0, 1.0, 1.0),
        )

        front_ortho = renderer.render(mesh, "front", ortho)
        front_perspective = renderer.render(mesh, "front", perspective)

        self.assertGreater(float(np.abs(front_ortho - front_perspective).sum()), 10.0)

    def test_rejects_empty_view_selection(self):
        renderer = MeshRenderer()
        with self.assertRaisesRegex(ValueError, "Select at least one"):
            renderer.render_views(make_box(), [], RenderSettings(width=32, height=32))

    def test_contact_sheet_builds_labeled_three_by_two_matrix(self):
        images = []
        labels = []
        for index, side in enumerate(["front", "back", "left", "right", "top", "bottom"]):
            image = np.zeros((8, 8, 3), dtype=np.float32)
            image[:, :, index % 3] = 0.25 + index * 0.1
            images.append(image)
            labels.append(side)

        sheet = ContactSheetBuilder().build(images, labels, ContactSheetSettings(layout="3x2", label_views=True))

        self.assertEqual(sheet.shape, (16, 24, 3))
        self.assertGreater(float(sheet[:8, :8].sum()), 0.0)
        self.assertGreater(float(sheet[8:16, 16:24].sum()), 0.0)

    def test_z_up_side_view_keeps_z_axis_upright(self):
        renderer = MeshRenderer()
        mesh = MeshBatchItem(
            vertices=np.array(
                [
                    [0.0, 0.0, -1.0],
                    [0.0, 0.0, 1.0],
                    [0.2, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            faces=np.array([[0, 1, 2]], dtype=np.int64),
        )
        vertices, _, _ = renderer._prepare_mesh(mesh)
        projected, *_ = renderer._project(
            vertices,
            view_pose("right", "z_up"),
            RenderSettings(width=64, height=64, up_axis="z_up", orthographic_scale=2.0),
        )

        self.assertLess(projected[1, 1], projected[0, 1])

    def test_accepts_trimesh_like_objects(self):
        FakeTrimesh = type("Trimesh", (), {"__module__": "trimesh.base"})
        fake = FakeTrimesh()
        fake.vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        fake.faces = np.array([[0, 1, 2]], dtype=np.int64)

        items = _mesh_batch_items(fake)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].vertices.shape, (3, 3))
        self.assertEqual(items[0].faces.shape, (1, 3))

    def test_file_type_detection_ignores_string_format_method(self):
        self.assertIsNone(_file_type_from_model("mesh.obj"))

        fake_file = type("FakeFile3D", (), {})()
        fake_file.format = "glb"
        self.assertEqual(_file_type_from_model(fake_file), "glb")

    def test_face_limit_downsamples_before_rasterization(self):
        faces = np.arange(30, dtype=np.int64).reshape(10, 3)
        item = MeshBatchItem(vertices=np.zeros((30, 3), dtype=np.float32), faces=faces)

        limited = _limit_faces(item, max_faces=4)

        self.assertEqual(limited.faces.shape, (4, 3))
        np.testing.assert_array_equal(limited.faces, faces[[0, 2, 5, 7]])

    @unittest.skipUnless(importlib.util.find_spec("trimesh"), "trimesh is not installed")
    def test_accepts_string_model_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "triangle.obj"
            path.write_text(
                "\n".join(
                    [
                        "v 0 0 0",
                        "v 1 0 0",
                        "v 0 1 0",
                        "f 1 2 3",
                    ]
                ),
                encoding="ascii",
            )

            items = _mesh_batch_items(str(path))

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].vertices.shape, (3, 3))
        self.assertEqual(items[0].faces.shape, (1, 3))


if __name__ == "__main__":
    unittest.main()
