from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

import numpy as np


VIEW_ORDER = ("front", "back", "left", "right", "top", "bottom")


class CameraMode(str, Enum):
    ORTHOGRAPHIC = "orthographic"
    PERSPECTIVE = "perspective"


@dataclass(frozen=True)
class CameraPose:
    name: str
    direction: np.ndarray
    up: np.ndarray


@dataclass(frozen=True)
class RenderSettings:
    width: int = 512
    height: int = 512
    camera_mode: CameraMode = CameraMode.ORTHOGRAPHIC
    up_axis: str = "-Y"
    front_axis: str | None = "+Z"
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mesh_color: tuple[float, float, float] = (0.82, 0.84, 0.88)
    fov_degrees: float = 45.0
    orthographic_scale: float = 1.2
    camera_distance: float = 2.4
    shading: bool = True
    ambient: float = 0.28


@dataclass(frozen=True)
class MeshBatchItem:
    vertices: np.ndarray
    faces: np.ndarray
    vertex_colors: np.ndarray | None = None


@dataclass(frozen=True)
class RenderedView:
    name: str
    image: np.ndarray


@dataclass(frozen=True)
class ContactSheetSettings:
    layout: str = "3x2"
    label_views: bool = True
    label_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    label_background: tuple[float, float, float] = (0.0, 0.0, 0.0)


AXES = ("+Z", "-Z", "+Y", "-Y", "+X", "-X")
AXIS_VECTORS = {
    "+Z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    "-Z": np.array([0.0, 0.0, -1.0], dtype=np.float32),
    "+Y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "-Y": np.array([0.0, -1.0, 0.0], dtype=np.float32),
    "+X": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "-X": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
}
OPPOSITE_AXES = {
    "+Z": "-Z",
    "-Z": "+Z",
    "+Y": "-Y",
    "-Y": "+Y",
    "+X": "-X",
    "-X": "+X",
}
DEFAULT_FRONT_AXIS_BY_UP_AXIS = {
    "+Z": "-Y",
    "-Z": "+Y",
    "+Y": "+Z",
    "-Y": "+Z",
    "+X": "-Z",
    "-X": "+Z",
}


def canonical_axis(axis: str, *, kind: str = "axis") -> str:
    aliases = {
        "z_up": "+Z",
        "+z": "+Z",
        "z": "+Z",
        "-z": "-Z",
        "y_up": "+Y",
        "+y": "+Y",
        "y": "+Y",
        "-y": "-Y",
        "x_up": "+X",
        "+x": "+X",
        "x": "+X",
        "-x": "-X",
    }
    text = str(axis).strip()
    canonical = aliases.get(text.lower(), text.upper())
    if canonical not in AXIS_VECTORS:
        raise ValueError(f"Unknown {kind}: {axis}")
    return canonical


def canonical_up_axis(up_axis: str) -> str:
    return canonical_axis(up_axis, kind="up axis")


def canonical_front_axis(front_axis: str) -> str:
    return canonical_axis(front_axis, kind="front axis")


def opposite_axis(axis: str) -> str:
    return OPPOSITE_AXES[canonical_axis(axis)]


def valid_front_axes(up_axis: str) -> list[str]:
    up = canonical_up_axis(up_axis)
    excluded = {up, OPPOSITE_AXES[up]}
    return [axis for axis in AXES if axis not in excluded]


def default_front_axis(up_axis: str) -> str:
    return DEFAULT_FRONT_AXIS_BY_UP_AXIS[canonical_up_axis(up_axis)]


def resolve_front_axis(up_axis: str, front_axis: str | None = None) -> str:
    up = canonical_up_axis(up_axis)
    if front_axis is None or str(front_axis).strip().lower() in {"", "auto", "default"}:
        front = default_front_axis(up)
    else:
        front = canonical_front_axis(front_axis)
    if front not in valid_front_axes(up):
        valid = ", ".join(valid_front_axes(up))
        raise ValueError(f"front_axis {front} is invalid for up_axis {up}; choose one of: {valid}")
    return front


def axis_vector(axis: str) -> np.ndarray:
    return AXIS_VECTORS[canonical_axis(axis)].copy()


def view_pose(view: str, up_axis: str = "+Z", front_axis: str | None = None) -> CameraPose:
    if view not in VIEW_ORDER:
        raise ValueError(f"Unknown view: {view}")

    up_name = canonical_up_axis(up_axis)
    front_name = resolve_front_axis(up_name, front_axis)
    up = axis_vector(up_name)
    front = axis_vector(front_name)
    right = np.cross(up, front).astype(np.float32)
    top_up = (-front).astype(np.float32)

    directions = {
        "front": front,
        "back": -front,
        "left": -right,
        "right": right,
        "top": up,
        "bottom": -up,
    }
    up_hints = {
        "front": up,
        "back": up,
        "left": up,
        "right": up,
        "top": top_up,
        "bottom": top_up,
    }
    return CameraPose(view, directions[view].astype(np.float32), up_hints[view].astype(np.float32))


VIEW_POSES = {name: view_pose(name) for name in VIEW_ORDER}


def parse_color(value: str | Sequence[float], fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("#") and len(text) == 7:
            try:
                return (
                    int(text[1:3], 16) / 255.0,
                    int(text[3:5], 16) / 255.0,
                    int(text[5:7], 16) / 255.0,
                )
            except ValueError:
                return fallback
        pieces = [piece.strip() for piece in text.split(",")]
        if len(pieces) == 3:
            try:
                values = tuple(float(piece) for piece in pieces)
            except ValueError:
                return fallback
            scale = 255.0 if max(values) > 1.0 else 1.0
            return tuple(_clamp(channel / scale) for channel in values)  # type: ignore[return-value]
        return fallback

    if len(value) < 3:
        return fallback
    scale = 255.0 if max(value[:3]) > 1.0 else 1.0
    return tuple(_clamp(float(channel) / scale) for channel in value[:3])  # type: ignore[return-value]


def selected_view_names(
    *,
    front: bool,
    back: bool,
    left: bool,
    right: bool,
    top: bool,
    bottom: bool,
) -> list[str]:
    selected = {
        "front": front,
        "back": back,
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
    }
    return [name for name in VIEW_ORDER if selected[name]]


class MeshRenderer:
    def render_views(
        self,
        mesh: MeshBatchItem,
        views: Iterable[str],
        settings: RenderSettings,
    ) -> list[RenderedView]:
        view_names = list(views)
        if not view_names:
            raise ValueError("Select at least one render side.")
        return [RenderedView(name, self.render(mesh, name, settings)) for name in view_names]

    def render(self, mesh: MeshBatchItem, view: str, settings: RenderSettings) -> np.ndarray:
        vertices, faces, vertex_colors = self._prepare_mesh(mesh)
        pose = view_pose(view, settings.up_axis, settings.front_axis)
        projected, depths, camera_vertices, world_vertices, light_direction = self._project(vertices, pose, settings)

        image = np.zeros((settings.height, settings.width, 3), dtype=np.float32)
        image[:, :] = np.asarray(settings.background_color, dtype=np.float32)
        z_buffer = np.full((settings.height, settings.width), np.inf, dtype=np.float32)

        base_color = np.asarray(settings.mesh_color, dtype=np.float32)
        for face in faces:
            face = face.astype(np.int64, copy=False)
            tri_2d = projected[face]
            tri_depth = depths[face]
            tri_camera = camera_vertices[face]
            tri_world = world_vertices[face]
            if np.any(tri_depth <= 0.0):
                continue
            if self._is_degenerate(tri_2d):
                continue

            face_color = self._face_color(base_color, vertex_colors, face)
            if settings.shading:
                shade = self._shade_triangle(tri_world, light_direction, settings.ambient)
                face_color = face_color * shade

            self._rasterize_triangle(image, z_buffer, tri_2d, tri_depth, face_color)

        return np.clip(image, 0.0, 1.0)

    def _prepare_mesh(self, mesh: MeshBatchItem) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"vertices must be an Nx3 array, got {vertices.shape}")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"faces must be an Mx3 array, got {faces.shape}")
        if vertices.shape[0] == 0 or faces.shape[0] == 0:
            raise ValueError("Mesh must contain vertices and faces.")

        vertices = np.nan_to_num(vertices, copy=True)
        valid_faces = np.all((faces >= 0) & (faces < vertices.shape[0]), axis=1)
        faces = faces[valid_faces]
        if faces.shape[0] == 0:
            raise ValueError("Mesh does not contain any valid triangular faces.")

        vertex_colors = None
        if mesh.vertex_colors is not None:
            vertex_colors = np.asarray(mesh.vertex_colors, dtype=np.float32)
            if vertex_colors.ndim == 2 and vertex_colors.shape[0] == vertices.shape[0]:
                vertex_colors = vertex_colors[:, :3]
                if vertex_colors.size and float(np.nanmax(vertex_colors)) > 1.0:
                    vertex_colors = vertex_colors / 255.0
                vertex_colors = np.clip(np.nan_to_num(vertex_colors), 0.0, 1.0)
            else:
                vertex_colors = None

        minimum = vertices.min(axis=0)
        maximum = vertices.max(axis=0)
        center = (minimum + maximum) * 0.5
        extent = float(np.max(maximum - minimum))
        if extent <= 1e-8:
            extent = 1.0
        vertices = (vertices - center) / extent
        return vertices.astype(np.float32), faces.astype(np.int64), vertex_colors

    def _project(
        self,
        vertices: np.ndarray,
        pose: CameraPose,
        settings: RenderSettings,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        direction = _normalize(pose.direction)
        up_hint = _normalize(pose.up)
        position = direction * max(float(settings.camera_distance), 0.25)
        forward = _normalize(-direction)
        right = _normalize(np.cross(forward, up_hint))
        up = _normalize(np.cross(right, forward))

        relative = vertices - position
        x_camera = relative @ right
        y_camera = relative @ up
        z_camera = relative @ forward
        aspect = settings.width / max(settings.height, 1)

        if settings.camera_mode == CameraMode.PERSPECTIVE:
            tangent = np.tan(np.radians(np.clip(settings.fov_degrees, 1.0, 170.0)) * 0.5)
            safe_depth = np.maximum(z_camera, 1e-4)
            x_ndc = x_camera / (safe_depth * tangent * aspect)
            y_ndc = y_camera / (safe_depth * tangent)
        else:
            vertical_span = max(float(settings.orthographic_scale), 0.05)
            horizontal_span = vertical_span * aspect
            x_ndc = x_camera / (horizontal_span * 0.5)
            y_ndc = y_camera / (vertical_span * 0.5)

        projected = np.stack(
            [
                (x_ndc + 1.0) * 0.5 * (settings.width - 1),
                (1.0 - (y_ndc + 1.0) * 0.5) * (settings.height - 1),
            ],
            axis=1,
        ).astype(np.float32)
        camera_vertices = np.stack([x_camera, y_camera, z_camera], axis=1).astype(np.float32)
        return projected, z_camera.astype(np.float32), camera_vertices, vertices, direction

    def _rasterize_triangle(
        self,
        image: np.ndarray,
        z_buffer: np.ndarray,
        tri_2d: np.ndarray,
        tri_depth: np.ndarray,
        color: np.ndarray,
    ) -> None:
        height, width = z_buffer.shape
        min_x = max(int(np.floor(float(np.min(tri_2d[:, 0])))), 0)
        max_x = min(int(np.ceil(float(np.max(tri_2d[:, 0])))), width - 1)
        min_y = max(int(np.floor(float(np.min(tri_2d[:, 1])))), 0)
        max_y = min(int(np.ceil(float(np.max(tri_2d[:, 1])))), height - 1)
        if min_x > max_x or min_y > max_y:
            return

        xs = np.arange(min_x, max_x + 1, dtype=np.float32) + 0.5
        ys = np.arange(min_y, max_y + 1, dtype=np.float32) + 0.5
        grid_x, grid_y = np.meshgrid(xs, ys)

        x0, y0 = tri_2d[0]
        x1, y1 = tri_2d[1]
        x2, y2 = tri_2d[2]
        denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(float(denominator)) <= 1e-8:
            return

        w0 = ((y1 - y2) * (grid_x - x2) + (x2 - x1) * (grid_y - y2)) / denominator
        w1 = ((y2 - y0) * (grid_x - x2) + (x0 - x2) * (grid_y - y2)) / denominator
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5)
        if not np.any(inside):
            return

        depth = w0 * tri_depth[0] + w1 * tri_depth[1] + w2 * tri_depth[2]
        region_z = z_buffer[min_y : max_y + 1, min_x : max_x + 1]
        update = inside & (depth < region_z)
        if not np.any(update):
            return

        region_z[update] = depth[update]
        region_image = image[min_y : max_y + 1, min_x : max_x + 1]
        region_image[update] = color

    def _shade_triangle(
        self,
        tri_world: np.ndarray,
        light_direction: np.ndarray,
        ambient: float,
    ) -> float:
        edge_a = tri_world[1] - tri_world[0]
        edge_b = tri_world[2] - tri_world[0]
        normal = _normalize(np.cross(edge_a, edge_b))
        if not np.all(np.isfinite(normal)):
            return 1.0
        diffuse = abs(float(np.dot(normal, light_direction)))
        ambient = _clamp(float(ambient))
        return ambient + (1.0 - ambient) * diffuse

    def _face_color(
        self,
        base_color: np.ndarray,
        vertex_colors: np.ndarray | None,
        face: np.ndarray,
    ) -> np.ndarray:
        if vertex_colors is None:
            return base_color.copy()
        return np.mean(vertex_colors[face], axis=0).astype(np.float32)

    def _is_degenerate(self, tri_2d: np.ndarray) -> bool:
        area = (
            (tri_2d[1, 0] - tri_2d[0, 0]) * (tri_2d[2, 1] - tri_2d[0, 1])
            - (tri_2d[2, 0] - tri_2d[0, 0]) * (tri_2d[1, 1] - tri_2d[0, 1])
        )
        return abs(float(area)) <= 1e-5


class ContactSheetBuilder:
    def label_images(
        self,
        images: Sequence[np.ndarray],
        labels: Sequence[str],
        settings: ContactSheetSettings | None = None,
    ) -> list[np.ndarray]:
        settings = settings or ContactSheetSettings()
        prepared = [self._prepare_image(image) for image in images]
        if not settings.label_views:
            return prepared
        return [self._draw_label(image, str(label), settings) for image, label in zip(prepared, labels)]

    def build(
        self,
        images: Sequence[np.ndarray],
        labels: Sequence[str],
        settings: ContactSheetSettings | None = None,
    ) -> np.ndarray:
        if not images:
            raise ValueError("No images were provided for the contact sheet.")
        settings = settings or ContactSheetSettings()
        prepared = [self._prepare_image(image) for image in images]
        tile_height, tile_width = prepared[0].shape[:2]
        if any(image.shape[:2] != (tile_height, tile_width) for image in prepared):
            raise ValueError("All contact sheet images must have the same size.")

        rows, cols = self._grid(len(prepared), settings.layout)
        sheet = np.zeros((rows * tile_height, cols * tile_width, 3), dtype=np.float32)
        for index, image in enumerate(prepared):
            row = index // cols
            col = index % cols
            y0 = row * tile_height
            x0 = col * tile_width
            sheet[y0 : y0 + tile_height, x0 : x0 + tile_width, :] = image

        if settings.label_views:
            sheet = self._draw_labels(sheet, labels, cols, tile_width, tile_height, settings)
        return np.clip(sheet, 0.0, 1.0)

    def _grid(self, count: int, layout: str) -> tuple[int, int]:
        if layout == "auto":
            cols = int(np.ceil(np.sqrt(count)))
        else:
            try:
                cols_text, _rows_text = layout.lower().split("x", 1)
                cols = int(cols_text)
            except Exception as exc:
                raise ValueError(f"Unknown matrix layout: {layout}") from exc
        cols = max(cols, 1)
        return int(np.ceil(count / cols)), cols

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError(f"Contact sheet image must be HxWx3, got {arr.shape}")
        return np.clip(arr[:, :, :3], 0.0, 1.0)

    def _draw_labels(
        self,
        sheet: np.ndarray,
        labels: Sequence[str],
        cols: int,
        tile_width: int,
        tile_height: int,
        settings: ContactSheetSettings,
    ) -> np.ndarray:
        try:
            from PIL import Image, ImageDraw, ImageFont
        except Exception as exc:
            raise RuntimeError("Pillow is required to draw contact sheet labels.") from exc

        pil = Image.fromarray((sheet * 255.0).round().astype(np.uint8))
        draw = ImageDraw.Draw(pil)
        font = self._font(ImageFont, tile_width, tile_height)
        text_color = tuple(int(_clamp(channel) * 255) for channel in settings.label_color)
        bg_color = tuple(int(_clamp(channel) * 255) for channel in settings.label_background)
        for index, label in enumerate(labels):
            row = index // cols
            col = index % cols
            padding = self._padding(tile_width, tile_height)
            x = col * tile_width + padding
            y = row * tile_height + padding
            text = str(label)
            bbox = draw.textbbox((x, y), text, font=font)
            margin = max(3, padding // 2)
            draw.rectangle((bbox[0] - margin, bbox[1] - margin, bbox[2] + margin, bbox[3] + margin), fill=bg_color)
            draw.text((x, y), text, fill=text_color, font=font)
        return np.asarray(pil, dtype=np.float32) / 255.0

    def _draw_label(self, image: np.ndarray, label: str, settings: ContactSheetSettings) -> np.ndarray:
        try:
            from PIL import Image, ImageDraw, ImageFont
        except Exception as exc:
            raise RuntimeError("Pillow is required to draw image labels.") from exc

        height, width = image.shape[:2]
        pil = Image.fromarray((image * 255.0).round().astype(np.uint8))
        draw = ImageDraw.Draw(pil)
        font = self._font(ImageFont, width, height)
        padding = self._padding(width, height)
        text_color = tuple(int(_clamp(channel) * 255) for channel in settings.label_color)
        bg_color = tuple(int(_clamp(channel) * 255) for channel in settings.label_background)
        bbox = draw.textbbox((padding, padding), label, font=font)
        margin = max(3, padding // 2)
        draw.rectangle((bbox[0] - margin, bbox[1] - margin, bbox[2] + margin, bbox[3] + margin), fill=bg_color)
        draw.text((padding, padding), label, fill=text_color, font=font)
        return np.asarray(pil, dtype=np.float32) / 255.0

    def _font(self, ImageFont, width: int, height: int):
        size = max(14, min(48, min(width, height) // 12))
        for name in ("DejaVuSans.ttf", "arial.ttf"):
            try:
                return ImageFont.truetype(name, size=size)
            except Exception:
                continue
        return ImageFont.load_default()

    def _padding(self, width: int, height: int) -> int:
        return max(6, min(width, height) // 48)


class NvdiffrastRenderer:
    def __init__(self, device: str | None = None) -> None:
        try:
            import torch
            import nvdiffrast.torch as dr
        except Exception as exc:
            raise RuntimeError("nvdiffrast backend requires torch and nvdiffrast to be installed.") from exc
        if device is None:
            if not torch.cuda.is_available():
                raise RuntimeError("nvdiffrast backend requires a CUDA-capable torch runtime.")
            device = "cuda"
        self._torch = torch
        self._dr = dr
        self._device = device
        self._ctx = dr.RasterizeCudaContext(device=device)
        self._cpu_renderer = MeshRenderer()

    def render_views(
        self,
        mesh: MeshBatchItem,
        views: Iterable[str],
        settings: RenderSettings,
    ) -> list[RenderedView]:
        view_names = list(views)
        if not view_names:
            raise ValueError("Select at least one render side.")
        return [RenderedView(name, self.render(mesh, name, settings)) for name in view_names]

    def render(self, mesh: MeshBatchItem, view: str, settings: RenderSettings) -> np.ndarray:
        vertices, faces, vertex_colors = self._cpu_renderer._prepare_mesh(mesh)
        pose = view_pose(view, settings.up_axis, settings.front_axis)
        projected, depths, _camera_vertices, world_vertices, light_direction = self._cpu_renderer._project(
            vertices, pose, settings
        )
        valid = np.all(depths[faces] > 0.0, axis=1)
        faces = faces[valid]
        if faces.shape[0] == 0:
            image = np.zeros((settings.height, settings.width, 3), dtype=np.float32)
            image[:, :] = np.asarray(settings.background_color, dtype=np.float32)
            return image

        x_ndc = projected[:, 0] / max(settings.width - 1, 1) * 2.0 - 1.0
        y_ndc = 1.0 - projected[:, 1] / max(settings.height - 1, 1) * 2.0
        depth_min = float(np.min(depths[faces]))
        depth_max = float(np.max(depths[faces]))
        depth_span = max(depth_max - depth_min, 1e-6)
        z_ndc = ((depths - depth_min) / depth_span) * 2.0 - 1.0
        clip = np.stack([x_ndc, y_ndc, z_ndc, np.ones_like(x_ndc)], axis=1).astype(np.float32)

        torch = self._torch
        dr = self._dr
        vertices_clip = torch.from_numpy(clip).to(self._device).unsqueeze(0).contiguous()
        faces_t = torch.from_numpy(faces.astype(np.int32, copy=False)).to(self._device).contiguous()
        rast, _ = dr.rasterize(self._ctx, vertices_clip, faces_t, (settings.height, settings.width))
        mask = (rast[..., -1:] > 0).float()

        base_color = np.asarray(settings.mesh_color, dtype=np.float32)
        if vertex_colors is not None:
            colors_np = vertex_colors.astype(np.float32, copy=False)
            colors_t = torch.from_numpy(colors_np).to(self._device).unsqueeze(0).contiguous()
            color = dr.interpolate(colors_t, rast, faces_t)[0]
        else:
            color = torch.ones(
                (1, settings.height, settings.width, 3),
                dtype=torch.float32,
                device=self._device,
            ) * torch.tensor(base_color, dtype=torch.float32, device=self._device)

        if settings.shading:
            shade_np = _face_shades(world_vertices, faces, light_direction, settings.ambient)
            shade_t = torch.from_numpy(shade_np[:, None]).to(self._device).unsqueeze(0).contiguous()
            face_attr_t = torch.arange(faces.shape[0], dtype=torch.int32, device=self._device)
            face_attr_t = face_attr_t[:, None].repeat(1, 3).contiguous()
            color = color * dr.interpolate(shade_t, rast, face_attr_t)[0]

        background = torch.tensor(settings.background_color, dtype=torch.float32, device=self._device)
        image = color * mask + background.reshape(1, 1, 1, 3) * (1.0 - mask)
        image = dr.antialias(image, rast, vertices_clip, faces_t)
        return np.clip(image[0].detach().cpu().numpy(), 0.0, 1.0).astype(np.float32)


def _face_shades(
    vertices: np.ndarray,
    faces: np.ndarray,
    light_direction: np.ndarray,
    ambient: float,
) -> np.ndarray:
    triangles = vertices[faces]
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    normals = np.cross(edge_a, edge_b)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, np.maximum(lengths, 1e-12), out=np.zeros_like(normals), where=lengths > 0)
    diffuse = np.abs(normals @ _normalize(light_direction))
    ambient = _clamp(float(ambient))
    return (ambient + (1.0 - ambient) * diffuse).astype(np.float32)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def _clamp(value: float) -> float:
    return min(max(value, 0.0), 1.0)
