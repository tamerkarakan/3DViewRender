from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

try:
    from .renderer import (
        CameraMode,
        MeshBatchItem,
        MeshRenderer,
        RenderSettings,
        VIEW_ORDER,
        parse_color,
        selected_view_names,
    )
except ImportError:  # Allows running tests from this directory without package install.
    from renderer import (  # type: ignore
        CameraMode,
        MeshBatchItem,
        MeshRenderer,
        RenderSettings,
        VIEW_ORDER,
        parse_color,
        selected_view_names,
    )


try:
    from comfy_api.latest import ComfyExtension

    try:
        from comfy_api.latest import IO
    except ImportError:
        from comfy_api.latest import io as IO  # type: ignore

    COMFY_API_AVAILABLE = True
except Exception:
    ComfyExtension = None  # type: ignore
    IO = None  # type: ignore
    COMFY_API_AVAILABLE = False

try:
    from typing_extensions import override
except Exception:
    def override(func):  # type: ignore
        return func


NODE_ID = "T3DViewRenderSixSides"
DISPLAY_NAME = "3D View Render: Six Sides"
CATEGORY = "3d/render"
CAMERA_MODES = [CameraMode.ORTHOGRAPHIC.value, CameraMode.PERSPECTIVE.value]


def _build_settings(
    *,
    resolution: int,
    camera_mode: str,
    background_color: str,
    mesh_color: str,
    fov_degrees: float,
    orthographic_scale: float,
    camera_distance: float,
    shading: bool,
) -> RenderSettings:
    mode = CameraMode(camera_mode)
    base = RenderSettings(
        width=int(resolution),
        height=int(resolution),
        camera_mode=mode,
        fov_degrees=float(fov_degrees),
        orthographic_scale=float(orthographic_scale),
        camera_distance=float(camera_distance),
        shading=bool(shading),
    )
    return replace(
        base,
        background_color=parse_color(background_color, base.background_color),
        mesh_color=parse_color(mesh_color, base.mesh_color),
    )


def _render(
    *,
    mesh: Any,
    resolution: int,
    camera_mode: str,
    front: bool,
    back: bool,
    left: bool,
    right: bool,
    top: bool,
    bottom: bool,
    background_color: str,
    mesh_color: str,
    fov_degrees: float,
    orthographic_scale: float,
    camera_distance: float,
    shading: bool,
) -> tuple[Any, str]:
    settings = _build_settings(
        resolution=resolution,
        camera_mode=camera_mode,
        background_color=background_color,
        mesh_color=mesh_color,
        fov_degrees=fov_degrees,
        orthographic_scale=orthographic_scale,
        camera_distance=camera_distance,
        shading=shading,
    )
    views = selected_view_names(front=front, back=back, left=left, right=right, top=top, bottom=bottom)
    renderer = MeshRenderer()

    images: list[np.ndarray] = []
    labels: list[str] = []
    for batch_index, item in enumerate(_mesh_batch_items(mesh)):
        for rendered in renderer.render_views(item, views, settings):
            images.append(rendered.image.astype(np.float32, copy=False))
            labels.append(f"{batch_index}:{rendered.name}")

    if not images:
        raise ValueError("No renderable mesh items were found.")

    try:
        import torch
    except Exception as exc:
        raise RuntimeError("ComfyUI torch runtime is required to return IMAGE tensors.") from exc

    return torch.from_numpy(np.stack(images, axis=0)).float(), "\n".join(labels)


def _mesh_batch_items(mesh: Any) -> list[MeshBatchItem]:
    if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
        raise TypeError("Expected a ComfyUI MESH object with vertices and faces.")

    vertices = _to_numpy(mesh.vertices)
    faces = _to_numpy(mesh.faces).astype(np.int64, copy=False)
    vertex_colors = _to_numpy(getattr(mesh, "vertex_colors", None))
    vertex_counts = _to_numpy(getattr(mesh, "vertex_counts", None))
    face_counts = _to_numpy(getattr(mesh, "face_counts", None))

    if vertices.ndim == 2:
        vertices = vertices[None, ...]
    if faces.ndim == 2:
        faces = faces[None, ...]
    if vertex_colors is not None and vertex_colors.ndim == 2:
        vertex_colors = vertex_colors[None, ...]

    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError(f"mesh.vertices must have shape BxNx3 or Nx3, got {vertices.shape}")
    if faces.ndim != 3 or faces.shape[-1] != 3:
        raise ValueError(f"mesh.faces must have shape BxMx3 or Mx3, got {faces.shape}")

    batch_size = min(vertices.shape[0], faces.shape[0])
    items: list[MeshBatchItem] = []
    for index in range(batch_size):
        vertex_count = _count_at(vertex_counts, index, vertices.shape[1])
        face_count = _count_at(face_counts, index, faces.shape[1])
        colors = None
        if vertex_colors is not None and index < vertex_colors.shape[0]:
            colors = vertex_colors[index, :vertex_count]
        items.append(
            MeshBatchItem(
                vertices=vertices[index, :vertex_count],
                faces=faces[index, :face_count],
                vertex_colors=colors,
            )
        )
    return items


def _to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _count_at(counts: np.ndarray | None, index: int, fallback: int) -> int:
    if counts is None:
        return fallback
    flat = np.asarray(counts).reshape(-1)
    if index >= flat.shape[0]:
        return fallback
    return max(0, min(int(flat[index]), fallback))


def _legacy_inputs() -> dict[str, dict[str, Any]]:
    return {
        "required": {
            "mesh": ("MESH",),
            "resolution": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
            "camera_mode": (CAMERA_MODES, {"default": CameraMode.ORTHOGRAPHIC.value}),
            "front": ("BOOLEAN", {"default": True}),
            "back": ("BOOLEAN", {"default": True}),
            "left": ("BOOLEAN", {"default": True}),
            "right": ("BOOLEAN", {"default": True}),
            "top": ("BOOLEAN", {"default": True}),
            "bottom": ("BOOLEAN", {"default": True}),
            "background_color": ("STRING", {"default": "#000000", "advanced": True}),
            "mesh_color": ("STRING", {"default": "#D1D5DB", "advanced": True}),
            "fov_degrees": ("FLOAT", {"default": 45.0, "min": 10.0, "max": 120.0, "step": 1.0, "advanced": True}),
            "orthographic_scale": ("FLOAT", {"default": 1.2, "min": 0.05, "max": 10.0, "step": 0.05, "advanced": True}),
            "camera_distance": ("FLOAT", {"default": 2.4, "min": 0.25, "max": 20.0, "step": 0.05, "advanced": True}),
            "shading": ("BOOLEAN", {"default": True, "advanced": True}),
        }
    }


if COMFY_API_AVAILABLE:

    class SixSideRender(IO.ComfyNode):  # type: ignore[misc]
        @classmethod
        def define_schema(cls):
            return IO.Schema(
                node_id=NODE_ID,
                display_name=DISPLAY_NAME,
                category=CATEGORY,
                search_aliases=[
                    "six side render",
                    "6 side render",
                    "orthographic mesh render",
                    "perspective mesh render",
                ],
                description="Render selected front/back/left/right/top/bottom views from a ComfyUI MESH.",
                inputs=[
                    IO.Mesh.Input("mesh", tooltip="ComfyUI MESH to render."),
                    IO.Int.Input("resolution", default=512, min=64, max=4096, step=64),
                    IO.Combo.Input("camera_mode", options=CAMERA_MODES, default=CameraMode.ORTHOGRAPHIC.value),
                    IO.Boolean.Input("front", default=True, label_on="render", label_off="skip"),
                    IO.Boolean.Input("back", default=True, label_on="render", label_off="skip"),
                    IO.Boolean.Input("left", default=True, label_on="render", label_off="skip"),
                    IO.Boolean.Input("right", default=True, label_on="render", label_off="skip"),
                    IO.Boolean.Input("top", default=True, label_on="render", label_off="skip"),
                    IO.Boolean.Input("bottom", default=True, label_on="render", label_off="skip"),
                    IO.String.Input("background_color", default="#000000", advanced=True),
                    IO.String.Input("mesh_color", default="#D1D5DB", advanced=True),
                    IO.Float.Input("fov_degrees", default=45.0, min=10.0, max=120.0, step=1.0, advanced=True),
                    IO.Float.Input("orthographic_scale", default=1.2, min=0.05, max=10.0, step=0.05, advanced=True),
                    IO.Float.Input("camera_distance", default=2.4, min=0.25, max=20.0, step=0.05, advanced=True),
                    IO.Boolean.Input("shading", default=True, advanced=True),
                ],
                outputs=[
                    IO.Image.Output(display_name="images"),
                    IO.String.Output(display_name="view_names"),
                ],
            )

        @classmethod
        def execute(
            cls,
            mesh,
            resolution: int = 512,
            camera_mode: str = CameraMode.ORTHOGRAPHIC.value,
            front: bool = True,
            back: bool = True,
            left: bool = True,
            right: bool = True,
            top: bool = True,
            bottom: bool = True,
            background_color: str = "#000000",
            mesh_color: str = "#D1D5DB",
            fov_degrees: float = 45.0,
            orthographic_scale: float = 1.2,
            camera_distance: float = 2.4,
            shading: bool = True,
        ):
            images, names = _render(
                mesh=mesh,
                resolution=resolution,
                camera_mode=camera_mode,
                front=front,
                back=back,
                left=left,
                right=right,
                top=top,
                bottom=bottom,
                background_color=background_color,
                mesh_color=mesh_color,
                fov_degrees=fov_degrees,
                orthographic_scale=orthographic_scale,
                camera_distance=camera_distance,
                shading=shading,
            )
            return IO.NodeOutput(images, names)

        render = execute


    class ThreeDViewRenderExtension(ComfyExtension):  # type: ignore[misc]
        @override
        async def get_node_list(self) -> list[type[IO.ComfyNode]]:  # type: ignore[name-defined]
            return [SixSideRender]


    async def comfy_entrypoint() -> ThreeDViewRenderExtension:
        return ThreeDViewRenderExtension()

else:

    class SixSideRender:
        RETURN_TYPES = ("IMAGE", "STRING")
        RETURN_NAMES = ("images", "view_names")
        FUNCTION = "render"
        CATEGORY = CATEGORY

        @classmethod
        def INPUT_TYPES(cls):
            return _legacy_inputs()

        def render(
            self,
            mesh,
            resolution: int = 512,
            camera_mode: str = CameraMode.ORTHOGRAPHIC.value,
            front: bool = True,
            back: bool = True,
            left: bool = True,
            right: bool = True,
            top: bool = True,
            bottom: bool = True,
            background_color: str = "#000000",
            mesh_color: str = "#D1D5DB",
            fov_degrees: float = 45.0,
            orthographic_scale: float = 1.2,
            camera_distance: float = 2.4,
            shading: bool = True,
        ):
            return _render(
                mesh=mesh,
                resolution=resolution,
                camera_mode=camera_mode,
                front=front,
                back=back,
                left=left,
                right=right,
                top=top,
                bottom=bottom,
                background_color=background_color,
                mesh_color=mesh_color,
                fov_degrees=fov_degrees,
                orthographic_scale=orthographic_scale,
                camera_distance=camera_distance,
                shading=shading,
            )
