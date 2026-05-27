from .nodes import COMFY_API_AVAILABLE, SixSideRender

WEB_DIRECTORY = "./web"

if COMFY_API_AVAILABLE:
    from .nodes import comfy_entrypoint

    __all__ = ["WEB_DIRECTORY", "comfy_entrypoint"]
else:
    NODE_CLASS_MAPPINGS = {
        "T3DViewRenderSixSides": SixSideRender,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "T3DViewRenderSixSides": "3D View Render: Six Sides",
    }

    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
