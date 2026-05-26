import subprocess
import sys
import textwrap
import unittest


class V3EntrypointContractTests(unittest.TestCase):
    def test_v3_schema_path_registers_entrypoint_without_legacy_mapping(self):
        script = textwrap.dedent(
            """
            import sys
            import types

            class Field:
                def __init__(self, id=None, display_name=None, **kwargs):
                    self.id = id
                    self.display_name = display_name
                    self.kwargs = kwargs

            class ComboInput(Field):
                def __init__(self, id, options=None, **kwargs):
                    super().__init__(id=id, **kwargs)
                    self.options = options

            def make_type(input_cls=Field):
                return type("T", (), {"Input": input_cls, "Output": Field})

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
            assert "camera_mode" in input_ids
            assert output_names == ["images", "view_names"]
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
