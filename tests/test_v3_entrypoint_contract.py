import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class V3EntrypointContractTests(unittest.TestCase):
    def test_v3_schema_path_registers_entrypoint_without_legacy_mapping(self):
        script = textwrap.dedent(
            """
            import sys
            import types
            from pathlib import Path

            sys.path.insert(0, str(Path.cwd()))

            class Field:
                def __init__(self, id=None, display_name=None, **kwargs):
                    self.id = id
                    self.display_name = display_name
                    self.kwargs = kwargs

            class ComboInput(Field):
                def __init__(self, id, options=None, **kwargs):
                    super().__init__(id=id, **kwargs)
                    self.options = options

            class MultiTypeInput(Field):
                def __init__(self, input_override, types=None, **kwargs):
                    super().__init__(id=input_override.id, **kwargs)
                    self.input_override = input_override
                    self.types = types or []

            def make_type(input_cls=Field):
                return type("T", (), {"Input": input_cls, "Output": Field})

            def custom_type(name):
                return type(name, (), {"io_type": name, "Input": Field, "Output": Field})

            class Schema:
                def __init__(self, **kwargs):
                    self.__dict__.update(kwargs)

            class NodeOutput:
                def __init__(self, *args, ui=None):
                    self.args = args
                    self.ui = ui

            class ComfyNode:
                pass

            class ComfyExtension:
                pass

            io = types.SimpleNamespace(
                ComfyNode=ComfyNode,
                Schema=Schema,
                NodeOutput=NodeOutput,
                Mesh=make_type(),
                Int=make_type(),
                Float=make_type(),
                String=make_type(),
                Boolean=make_type(),
                Image=make_type(),
                Combo=make_type(ComboInput),
                MultiType=type("MultiType", (), {"Input": MultiTypeInput}),
                Custom=custom_type,
                File3DGLB=custom_type("FILE_3D_GLB"),
                File3DGLTF=custom_type("FILE_3D_GLTF"),
                File3DOBJ=custom_type("FILE_3D_OBJ"),
                File3DFBX=custom_type("FILE_3D_FBX"),
                File3DSTL=custom_type("FILE_3D_STL"),
                File3DUSDZ=custom_type("FILE_3D_USDZ"),
                File3DAny=custom_type("FILE_3D"),
            )

            comfy_api = types.ModuleType("comfy_api")
            latest = types.ModuleType("comfy_api.latest")
            latest.ComfyExtension = ComfyExtension
            latest.IO = io
            sys.modules["comfy_api"] = comfy_api
            sys.modules["comfy_api.latest"] = latest

            import nodes

            assert nodes.COMFY_API_AVAILABLE is True
            assert hasattr(nodes, "comfy_entrypoint")
            schema = nodes.SixSideRender.define_schema()
            input_ids = [field.id for field in schema.inputs]
            output_names = [field.display_name for field in schema.outputs]

            assert schema.node_id == "T3DViewRenderSixSides"
            assert schema.category == "3d/render"
            for side in ("front", "back", "left", "right", "top", "bottom"):
                assert side in input_ids
            assert "model" in input_ids
            assert "renderer_backend" in input_ids
            assert "auto_install_f3d" in input_ids
            assert "blender_path" in input_ids
            assert "max_faces" in input_ids
            assert "camera_mode" in input_ids
            assert "up_axis" in input_ids
            assert "matrix_layout" in input_ids
            assert "label_matrix" in input_ids
            assert "save_to_output" in input_ids
            assert "filename_prefix" in input_ids
            assert schema.is_output_node is True
            assert output_names == ["images", "view_names", "render_info", "contact_sheet"]

            def fake_render(**kwargs):
                return "images", "right", "renderer_backend=cpu_preview", "contact_sheet", {"text": ["renderer_backend=cpu_preview"]}

            nodes._render = fake_render
            output = nodes.SixSideRender.execute(model="mesh")
            assert output.args == ("images", "right", "renderer_backend=cpu_preview", "contact_sheet")
            assert output.ui == {"text": ["renderer_backend=cpu_preview"]}
            """
        )

        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=".",
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
