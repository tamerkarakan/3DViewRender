from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .external_renderers import BlenderRenderer, F3DRenderer
except ImportError:
    from external_renderers import BlenderRenderer, F3DRenderer  # type: ignore

try:
    from .renderer import (
        CameraMode,
        MeshBatchItem,
        MeshRenderer,
        NvdiffrastRenderer,
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
        NvdiffrastRenderer,
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
RENDERER_BACKENDS = ["cpu_preview", "f3d_optional", "blender_optional", "nvdiffrast_optional"]
UP_AXES = ["z_up", "y_up"]
MODEL_INPUT_TYPES = (
    "MESH",
    "TRIMESH",
    "MESHWITHVOXEL",
    "FILE_3D_GLB",
    "FILE_3D_GLTF",
    "FILE_3D_OBJ",
    "FILE_3D_FBX",
    "FILE_3D_STL",
    "FILE_3D_USDZ",
    "FILE_3D",
    "STRING",
)


def _build_settings(
    *,
    resolution: int,
    camera_mode: str,
    background_color: str,
    mesh_color: str,
    up_axis: str,
    fov_degrees: float,
    orthographic_scale: float,
    camera_distance: float,
    shading: bool,
) -> RenderSettings:
    mode = CameraMode(camera_mode)
    if up_axis not in UP_AXES:
        raise ValueError(f"Unknown up axis: {up_axis}")
    base = RenderSettings(
        width=int(resolution),
        height=int(resolution),
        camera_mode=mode,
        up_axis=up_axis,
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
    model: Any,
    resolution: int,
    max_faces: int,
    renderer_backend: str,
    auto_install_f3d: bool,
    blender_path: str,
    camera_mode: str,
    front: bool,
    back: bool,
    left: bool,
    right: bool,
    top: bool,
    bottom: bool,
    background_color: str,
    mesh_color: str,
    up_axis: str,
    fov_degrees: float,
    orthographic_scale: float,
    camera_distance: float,
    shading: bool,
) -> tuple[Any, str, str]:
    settings = _build_settings(
        resolution=resolution,
        camera_mode=camera_mode,
        background_color=background_color,
        mesh_color=mesh_color,
        up_axis=up_axis,
        fov_degrees=fov_degrees,
        orthographic_scale=orthographic_scale,
        camera_distance=camera_distance,
        shading=shading,
    )
    views = selected_view_names(front=front, back=back, left=left, right=right, top=top, bottom=bottom)
    if renderer_backend in {"f3d_optional", "blender_optional"}:
        return _render_external(
            model=model,
            views=views,
            settings=settings,
            renderer_backend=renderer_backend,
            auto_install_f3d=auto_install_f3d,
            blender_path=blender_path,
        )

    if renderer_backend == "nvdiffrast_optional":
        renderer = NvdiffrastRenderer()
        mesh_max_faces = 0
    elif renderer_backend == "cpu_preview":
        renderer = MeshRenderer()
        mesh_max_faces = max_faces
    else:
        raise ValueError(f"Unknown renderer backend: {renderer_backend}")

    images: list[np.ndarray] = []
    labels: list[str] = []
    for batch_index, item in enumerate(_mesh_batch_items(model, max_faces=mesh_max_faces)):
        for rendered in renderer.render_views(item, views, settings):
            images.append(rendered.image.astype(np.float32, copy=False))
            labels.append(f"{batch_index}:{rendered.name}")

    if not images:
        raise ValueError("No renderable mesh items were found.")

    try:
        import torch
    except Exception as exc:
        raise RuntimeError("ComfyUI torch runtime is required to return IMAGE tensors.") from exc

    render_info = _format_render_info(
        renderer_backend=renderer_backend,
        settings=settings,
        views=views,
        max_faces=max_faces if renderer_backend == "cpu_preview" else 0,
    )
    return torch.from_numpy(np.stack(images, axis=0)).float(), "\n".join(labels), render_info


def _render_external(
    *,
    model: Any,
    views: list[str],
    settings: RenderSettings,
    renderer_backend: str,
    auto_install_f3d: bool,
    blender_path: str,
) -> tuple[Any, str, str]:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("ComfyUI torch runtime is required to return IMAGE tensors.") from exc

    paths, cleanup = _external_model_paths(model)
    try:
        if renderer_backend == "f3d_optional":
            renderer = F3DRenderer(auto_install=auto_install_f3d)
            blender_executable = None
        elif renderer_backend == "blender_optional":
            renderer = BlenderRenderer(blender_path=blender_path)
            blender_executable = renderer.blender_executable
        else:
            raise ValueError(f"Unknown external renderer backend: {renderer_backend}")

        images: list[np.ndarray] = []
        labels: list[str] = []
        for batch_index, path in enumerate(paths):
            for rendered in renderer.render_file_views(path, views, settings):
                images.append(rendered.image.astype(np.float32, copy=False))
                labels.append(f"{batch_index}:{rendered.name}")
        if not images:
            raise ValueError("No renderable mesh items were found.")
        render_info = _format_render_info(
            renderer_backend=renderer_backend,
            settings=settings,
            views=views,
            auto_install_f3d=auto_install_f3d,
            blender_executable=blender_executable,
        )
        return torch.from_numpy(np.stack(images, axis=0)).float(), "\n".join(labels), render_info
    finally:
        if cleanup is not None:
            cleanup.cleanup()


def _format_render_info(
    *,
    renderer_backend: str,
    settings: RenderSettings,
    views: list[str],
    max_faces: int | None = None,
    auto_install_f3d: bool | None = None,
    blender_executable: str | None = None,
) -> str:
    engine_labels = {
        "cpu_preview": "MeshRenderer CPU rasterizer",
        "f3d_optional": "F3D offscreen renderer",
        "blender_optional": "Blender background renderer",
        "nvdiffrast_optional": "nvdiffrast CUDA renderer",
    }
    lines = [
        f"renderer_backend={renderer_backend}",
        f"renderer_engine={engine_labels.get(renderer_backend, renderer_backend)}",
        f"camera_mode={settings.camera_mode.value}",
        f"up_axis={settings.up_axis}",
        f"resolution={settings.width}x{settings.height}",
        f"views={','.join(views)}",
    ]
    if renderer_backend == "cpu_preview":
        lines.append(f"max_faces={max_faces if max_faces and max_faces > 0 else 'unlimited'}")
    elif renderer_backend == "nvdiffrast_optional":
        lines.append("max_faces=unlimited")
    elif renderer_backend == "f3d_optional":
        lines.append(f"auto_install_f3d={str(bool(auto_install_f3d)).lower()}")
    elif renderer_backend == "blender_optional" and blender_executable:
        lines.append(f"blender_executable={blender_executable}")
    return "\n".join(lines)


def _render_info_ui(render_info: str) -> dict[str, list[str]]:
    return {"text": render_info.splitlines()}


def _external_model_paths(model: Any) -> tuple[list[str], tempfile.TemporaryDirectory[str] | None]:
    source = _file_like_source(model)
    if isinstance(source, str):
        return [_resolve_3d_path(source)], None

    temp_dir = tempfile.TemporaryDirectory(prefix="3dviewrender_model_")
    root = Path(temp_dir.name)
    if source is not None:
        suffix = f".{_file_type_from_model(model) or 'glb'}"
        path = root / f"model{suffix}"
        data = source if isinstance(source, (bytes, bytearray)) else bytes(source)
        path.write_bytes(data)
        return [str(path)], temp_dir

    try:
        import trimesh
    except Exception as exc:
        temp_dir.cleanup()
        raise RuntimeError("trimesh is required to export in-memory mesh inputs for external renderers.") from exc

    paths: list[str] = []
    try:
        for index, item in enumerate(_mesh_batch_items(model, max_faces=0)):
            path = root / f"mesh_{index}.obj"
            trimesh.Trimesh(vertices=item.vertices, faces=item.faces, process=False).export(path)
            paths.append(str(path))
    except Exception:
        temp_dir.cleanup()
        raise
    return paths, temp_dir


def _mesh_batch_items(mesh: Any, max_faces: int | None = None) -> list[MeshBatchItem]:
    from_file = _mesh_items_from_file_like(mesh, max_faces=max_faces)
    if from_file is not None:
        return from_file

    if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
        raise TypeError("Expected a mesh object, trimesh object, File3D, or 3D file path.")

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
            _limit_faces(
                MeshBatchItem(
                    vertices=vertices[index, :vertex_count],
                    faces=faces[index, :face_count],
                    vertex_colors=colors,
                ),
                max_faces=max_faces,
            )
        )
    return items


def _mesh_items_from_file_like(model: Any, max_faces: int | None = None) -> list[MeshBatchItem] | None:
    if _looks_like_trimesh(model):
        return _items_from_trimesh(model, max_faces=max_faces)

    source = _file_like_source(model)
    if source is None:
        return None

    try:
        import trimesh
    except Exception as exc:
        raise RuntimeError("trimesh is required to render FILE_3D or path inputs.") from exc

    file_type = _file_type_from_model(model)
    loaded = trimesh.load(source, file_type=file_type, force="scene")
    return _items_from_trimesh(loaded, max_faces=max_faces)


def _file_type_from_model(model: Any) -> str | None:
    if isinstance(model, str):
        return None
    file_type = getattr(model, "format", None)
    if isinstance(file_type, str) and file_type.strip():
        return file_type
    return None


def _file_like_source(model: Any) -> Any | None:
    if isinstance(model, str):
        text = model.strip()
        return _resolve_3d_path(text) if text else None
    if hasattr(model, "get_source"):
        source = model.get_source()
        if isinstance(source, str):
            return source
        return getattr(model, "get_data", lambda: source)()
    if hasattr(model, "get_data") and hasattr(model, "format"):
        return model.get_data()
    return None


def _resolve_3d_path(path: str) -> str:
    import os

    if os.path.isabs(path) or os.path.exists(path):
        return path
    try:
        import folder_paths

        roots = [
            folder_paths.get_output_directory(),
            folder_paths.get_input_directory(),
            folder_paths.get_temp_directory(),
        ]
        for root in roots:
            candidate = os.path.join(root, path)
            if os.path.exists(candidate):
                return candidate
    except Exception:
        pass
    return path


def _looks_like_trimesh(model: Any) -> bool:
    module = getattr(model.__class__, "__module__", "")
    name = getattr(model.__class__, "__name__", "")
    return (
        module.startswith("trimesh")
        or name in {"Trimesh", "Scene"}
        or (hasattr(model, "geometry") and hasattr(model, "dump"))
    )


def _items_from_trimesh(model: Any, max_faces: int | None = None) -> list[MeshBatchItem]:
    meshes = _flatten_trimesh(model)
    items = []
    for mesh in meshes:
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        colors = _trimesh_vertex_colors(mesh, vertices.shape[0])
        if vertices.size and faces.size:
            items.append(
                _limit_faces(
                    MeshBatchItem(vertices=vertices, faces=faces, vertex_colors=colors),
                    max_faces=max_faces,
                )
            )
    if not items:
        raise ValueError("The 3D input did not contain renderable triangular geometry.")
    return items


def _limit_faces(item: MeshBatchItem, max_faces: int | None) -> MeshBatchItem:
    if max_faces is None or max_faces <= 0 or item.faces.shape[0] <= max_faces:
        return item
    step = item.faces.shape[0] / float(max_faces)
    indices = np.floor(np.arange(max_faces, dtype=np.float64) * step).astype(np.int64)
    return MeshBatchItem(vertices=item.vertices, faces=item.faces[indices], vertex_colors=item.vertex_colors)


def _flatten_trimesh(model: Any) -> list[Any]:
    if hasattr(model, "geometry") and hasattr(model, "dump"):
        dumped = model.dump(concatenate=False)
        if isinstance(dumped, list):
            return dumped
        return [dumped]
    return [model]


def _trimesh_vertex_colors(mesh: Any, vertex_count: int) -> np.ndarray | None:
    visual = getattr(mesh, "visual", None)
    colors = getattr(visual, "vertex_colors", None)
    if colors is None:
        return None
    colors = np.asarray(colors)
    if colors.ndim != 2 or colors.shape[0] != vertex_count:
        return None
    return colors[:, :3]


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
            "model": (",".join(MODEL_INPUT_TYPES), {"tooltip": "MESH, TRIMESH, MESHWITHVOXEL, File3D, or 3D file path."}),
            "resolution": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
            "renderer_backend": (
                RENDERER_BACKENDS,
                {
                    "default": "cpu_preview",
                    "tooltip": "cpu_preview has no extra deps. f3d_optional is free/BSD and can auto-install. blender_optional uses a local Blender install.",
                },
            ),
            "auto_install_f3d": (
                "BOOLEAN",
                {
                    "default": False,
                    "advanced": True,
                    "tooltip": "When using f3d_optional, allow one-time pip install f3d if missing.",
                },
            ),
            "blender_path": (
                "STRING",
                {
                    "default": "",
                    "advanced": True,
                    "tooltip": "Optional Blender folder or blender.exe path. Leave blank to auto-detect PATH/common install folders.",
                },
            ),
            "max_faces": (
                "INT",
                {
                    "default": 10000,
                    "min": 0,
                    "max": 2000000,
                    "step": 1000,
                    "advanced": True,
                    "tooltip": "Caps faces before CPU rasterization. Set 0 to disable for small meshes only.",
                },
            ),
            "camera_mode": (CAMERA_MODES, {"default": CameraMode.ORTHOGRAPHIC.value}),
            "up_axis": (
                UP_AXES,
                {
                    "default": "z_up",
                    "tooltip": "Model vertical axis. z_up keeps side views upright for most GLB/OBJ assets; use y_up for Y-up models.",
                },
            ),
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

    def _model_input():
        return IO.MultiType.Input(
            IO.Mesh.Input("model", tooltip="MESH, TRIMESH, MESHWITHVOXEL, File3D, or 3D file path."),
            types=[
                IO.Custom("TRIMESH"),
                IO.Custom("MESHWITHVOXEL"),
                IO.File3DGLB,
                IO.File3DGLTF,
                IO.File3DOBJ,
                IO.File3DFBX,
                IO.File3DSTL,
                IO.File3DUSDZ,
                IO.File3DAny,
                IO.String,
            ],
        )


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
                description="Render selected front/back/left/right/top/bottom views from a ComfyUI 3D model.",
                inputs=[
                    _model_input(),
                    IO.Int.Input("resolution", default=512, min=64, max=4096, step=64),
                    IO.Combo.Input(
                        "renderer_backend",
                        options=RENDERER_BACKENDS,
                        default="cpu_preview",
                        tooltip="cpu_preview has no extra deps. f3d_optional is free/BSD and can auto-install. blender_optional uses a local Blender install.",
                    ),
                    IO.Boolean.Input(
                        "auto_install_f3d",
                        default=False,
                        advanced=True,
                        tooltip="When using f3d_optional, allow one-time pip install f3d if missing.",
                    ),
                    IO.String.Input(
                        "blender_path",
                        default="",
                        advanced=True,
                        tooltip="Optional Blender folder or blender.exe path. Leave blank to auto-detect PATH/common install folders.",
                    ),
                    IO.Int.Input(
                        "max_faces",
                        default=10000,
                        min=0,
                        max=2000000,
                        step=1000,
                        advanced=True,
                        tooltip="Caps faces before CPU rasterization. Set 0 to disable for small meshes only.",
                    ),
                    IO.Combo.Input("camera_mode", options=CAMERA_MODES, default=CameraMode.ORTHOGRAPHIC.value),
                    IO.Combo.Input(
                        "up_axis",
                        options=UP_AXES,
                        default="z_up",
                        tooltip="Model vertical axis. z_up keeps side views upright for most GLB/OBJ assets; use y_up for Y-up models.",
                    ),
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
                    IO.String.Output(display_name="render_info"),
                ],
            )

        @classmethod
        def execute(
            cls,
            model,
            resolution: int = 512,
            renderer_backend: str = "cpu_preview",
            auto_install_f3d: bool = False,
            blender_path: str = "",
            max_faces: int = 10000,
            camera_mode: str = CameraMode.ORTHOGRAPHIC.value,
            up_axis: str = "z_up",
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
            images, names, render_info = _render(
                model=model,
                resolution=resolution,
                renderer_backend=renderer_backend,
                auto_install_f3d=auto_install_f3d,
                blender_path=blender_path,
                max_faces=max_faces,
                camera_mode=camera_mode,
                up_axis=up_axis,
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
            return IO.NodeOutput(images, names, render_info, ui=_render_info_ui(render_info))

        render = execute


    class ThreeDViewRenderExtension(ComfyExtension):  # type: ignore[misc]
        @override
        async def get_node_list(self) -> list[type[IO.ComfyNode]]:  # type: ignore[name-defined]
            return [SixSideRender]


    async def comfy_entrypoint() -> ThreeDViewRenderExtension:
        return ThreeDViewRenderExtension()

else:

    class SixSideRender:
        RETURN_TYPES = ("IMAGE", "STRING", "STRING")
        RETURN_NAMES = ("images", "view_names", "render_info")
        FUNCTION = "render"
        CATEGORY = CATEGORY

        @classmethod
        def INPUT_TYPES(cls):
            return _legacy_inputs()

        def render(
            self,
            model,
            resolution: int = 512,
            renderer_backend: str = "cpu_preview",
            auto_install_f3d: bool = False,
            blender_path: str = "",
            max_faces: int = 10000,
            camera_mode: str = CameraMode.ORTHOGRAPHIC.value,
            up_axis: str = "z_up",
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
            images, names, render_info = _render(
                model=model,
                resolution=resolution,
                renderer_backend=renderer_backend,
                auto_install_f3d=auto_install_f3d,
                blender_path=blender_path,
                max_faces=max_faces,
                camera_mode=camera_mode,
                up_axis=up_axis,
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
            return {
                "ui": _render_info_ui(render_info),
                "result": (images, names, render_info),
            }
