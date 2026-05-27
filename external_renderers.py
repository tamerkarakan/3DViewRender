from __future__ import annotations

import importlib
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from .renderer import CameraMode, RenderSettings, RenderedView, VIEW_POSES
except ImportError:
    from renderer import CameraMode, RenderSettings, RenderedView, VIEW_POSES  # type: ignore


@dataclass(frozen=True)
class ModelBounds:
    center: np.ndarray
    extent: float


class F3DRenderer:
    def __init__(self, *, auto_install: bool = False) -> None:
        self._f3d = _import_f3d(auto_install=auto_install)

    def render_file_views(
        self,
        model_path: str,
        views: Iterable[str],
        settings: RenderSettings,
    ) -> list[RenderedView]:
        view_names = list(views)
        if not view_names:
            raise ValueError("Select at least one render side.")
        bounds = load_model_bounds(model_path)
        return [RenderedView(name, self._render_file(model_path, name, settings, bounds)) for name in view_names]

    def _render_file(
        self,
        model_path: str,
        view: str,
        settings: RenderSettings,
        bounds: ModelBounds,
    ) -> np.ndarray:
        if view not in VIEW_POSES:
            raise ValueError(f"Unknown view: {view}")
        f3d = self._f3d
        f3d.Engine.autoload_plugins()
        engine = f3d.Engine.create(True)
        engine.window.size = (int(settings.width), int(settings.height))
        engine.options["render.background.color"] = list(settings.background_color)
        engine.options["render.grid.enable"] = False
        engine.options["render.axes_grid.enable"] = False
        engine.options["ui.axis"] = False
        engine.options["ui.filename"] = False
        engine.options["ui.metadata"] = False
        engine.options["ui.loader_progress"] = False
        engine.options["render.effect.antialiasing.enable"] = True
        engine.scene.add(str(model_path))

        pose = VIEW_POSES[view]
        direction = _normalize(pose.direction)
        up = _normalize(pose.up)
        distance = max(bounds.extent * float(settings.camera_distance), bounds.extent * 1.5, 0.25)
        if settings.camera_mode == CameraMode.ORTHOGRAPHIC:
            # F3D's Python camera API exposes perspective FOV but not VTK parallel projection.
            # A narrow FOV from farther away keeps this backend visually close to orthographic.
            engine.window.camera.view_angle = 3.0
            distance = max(distance, bounds.extent / (2.0 * math.tan(math.radians(1.5))))
        else:
            engine.window.camera.view_angle = float(settings.fov_degrees)
        center = tuple(float(v) for v in bounds.center)
        position = tuple(float(v) for v in bounds.center + direction * distance)
        engine.window.camera.focal_point = center
        engine.window.camera.position = position
        engine.window.camera.view_up = tuple(float(v) for v in up)

        image = engine.window.render_to_image()
        return _f3d_image_to_array(image)


class BlenderRenderer:
    def __init__(self, *, blender_path: str = "") -> None:
        self._blender_executable = resolve_blender_executable(blender_path)

    @property
    def blender_executable(self) -> str:
        return self._blender_executable

    def render_file_views(
        self,
        model_path: str,
        views: Iterable[str],
        settings: RenderSettings,
    ) -> list[RenderedView]:
        view_names = list(views)
        if not view_names:
            raise ValueError("Select at least one render side.")

        with tempfile.TemporaryDirectory(prefix="3dviewrender_blender_") as temp_dir:
            temp_path = Path(temp_dir)
            config_path = temp_path / "render_config.json"
            script_path = temp_path / "render_script.py"
            output_dir = temp_path / "images"
            output_dir.mkdir()
            _write_blender_config(config_path, model_path, view_names, settings, output_dir)
            script_path.write_text(_BLENDER_RENDER_SCRIPT, encoding="utf-8")
            completed = subprocess.run(
                [
                    self._blender_executable,
                    "--background",
                    "--factory-startup",
                    "--python",
                    str(script_path),
                    "--",
                    str(config_path),
                ],
                text=True,
                capture_output=True,
                check=False,
                timeout=300,
            )
            if completed.returncode != 0:
                message = (completed.stderr or completed.stdout or "Blender render failed.").strip()
                raise RuntimeError(f"Blender render failed: {message[-2000:]}")

            from PIL import Image

            rendered: list[RenderedView] = []
            for name in view_names:
                png_path = output_dir / f"{name}.png"
                if not png_path.exists():
                    raise RuntimeError(f"Blender did not write expected render: {png_path}")
                arr = np.asarray(Image.open(png_path).convert("RGB"), dtype=np.float32) / 255.0
                rendered.append(RenderedView(name, arr))
            return rendered


def _import_f3d(*, auto_install: bool):
    try:
        return importlib.import_module("f3d")
    except ModuleNotFoundError as exc:
        if not auto_install:
            raise RuntimeError(
                "F3D backend requires the 'f3d' Python package. Enable auto_install_f3d "
                f"or run: {sys.executable} -m pip install f3d"
            ) from exc
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "install", "f3d"],
            text=True,
            capture_output=True,
            check=False,
            timeout=300,
        )
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or "pip install f3d failed.").strip()
            raise RuntimeError(f"Could not auto-install f3d: {message[-2000:]}") from exc
        importlib.invalidate_caches()
        return importlib.import_module("f3d")


def _f3d_image_to_array(image) -> np.ndarray:
    channel_count = int(image.channel_count)
    if channel_count < 3:
        raise RuntimeError(f"F3D returned an image with {channel_count} channels.")
    dtype = np.float32 if str(image.channel_type).endswith("FLOAT") else np.uint8
    arr = np.frombuffer(image.content, dtype=dtype)
    arr = arr.reshape((int(image.height), int(image.width), channel_count))[:, :, :3]
    if dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return np.clip(arr.astype(np.float32), 0.0, 1.0)


def load_model_bounds(model_path: str) -> ModelBounds:
    try:
        import trimesh

        loaded = trimesh.load(model_path, force="scene")
        meshes = loaded.dump(concatenate=False) if hasattr(loaded, "dump") else [loaded]
        bounds = [np.asarray(mesh.bounds, dtype=np.float32) for mesh in meshes if hasattr(mesh, "bounds")]
        bounds = [bound for bound in bounds if bound.shape == (2, 3) and np.all(np.isfinite(bound))]
        if bounds:
            minimum = np.min(np.stack([bound[0] for bound in bounds]), axis=0)
            maximum = np.max(np.stack([bound[1] for bound in bounds]), axis=0)
            center = (minimum + maximum) * 0.5
            extent = float(np.max(maximum - minimum))
            return ModelBounds(center=center.astype(np.float32), extent=max(extent, 1.0))
    except Exception:
        pass
    return ModelBounds(center=np.zeros(3, dtype=np.float32), extent=1.0)


def resolve_blender_executable(path: str = "") -> str:
    candidates = []
    text = path.strip().strip('"')
    if text:
        candidates.extend(_blender_candidates_from_path(Path(text)))
    for env_name in ("BLENDER_PATH", "BLENDER_EXE"):
        value = os.environ.get(env_name, "").strip().strip('"')
        if value:
            candidates.extend(_blender_candidates_from_path(Path(value)))
    which = shutil.which("blender")
    if which:
        candidates.append(Path(which))
    candidates.extend(_common_blender_candidates())

    seen = set()
    for candidate in candidates:
        normalized = str(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError(
        "Blender backend requires Blender. Set blender_path to a Blender folder or blender.exe path, "
        "or add Blender to PATH."
    )


def _blender_candidates_from_path(path: Path) -> list[Path]:
    if path.name.lower() == "blender.exe":
        return [path]
    return [path / "blender.exe", path / "Blender" / "blender.exe"]


def _common_blender_candidates() -> list[Path]:
    candidates: list[Path] = []
    roots = [
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")),
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")),
        Path(os.environ.get("LOCALAPPDATA", "")),
    ]
    for root in roots:
        if not root:
            continue
        try:
            candidates.extend(root.glob("Blender Foundation/Blender*/blender.exe"))
            candidates.extend(root.glob("Blender*/blender.exe"))
        except OSError:
            continue
    return candidates


def _write_blender_config(
    config_path: Path,
    model_path: str,
    view_names: list[str],
    settings: RenderSettings,
    output_dir: Path,
) -> None:
    import json

    config = {
        "model_path": str(model_path),
        "views": view_names,
        "output_dir": str(output_dir),
        "resolution": int(settings.width),
        "camera_mode": settings.camera_mode.value,
        "background_color": list(settings.background_color),
        "mesh_color": list(settings.mesh_color),
        "fov_degrees": float(settings.fov_degrees),
        "orthographic_scale": float(settings.orthographic_scale),
        "camera_distance": float(settings.camera_distance),
        "shading": bool(settings.shading),
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


_BLENDER_RENDER_SCRIPT = r'''
import json
import math
import os
import sys

import bpy
from mathutils import Matrix, Vector


VIEW_POSES = {
    "front": ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    "back": ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    "left": ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "right": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "top": ((0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
    "bottom": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
}


def import_model(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=path)
        else:
            bpy.ops.import_scene.obj(filepath=path)
    elif ext == ".stl":
        if hasattr(bpy.ops.wm, "stl_import"):
            bpy.ops.wm.stl_import(filepath=path)
        else:
            bpy.ops.import_mesh.stl(filepath=path)
    elif ext == ".ply":
        bpy.ops.wm.ply_import(filepath=path)
    else:
        raise RuntimeError(f"Unsupported Blender import format: {ext}")


def mesh_objects():
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def normalize_scene():
    bpy.context.view_layer.update()
    objects = mesh_objects()
    if not objects:
        raise RuntimeError("No mesh objects imported.")
    points = []
    for obj in objects:
        for corner in obj.bound_box:
            points.append(obj.matrix_world @ Vector(corner))
    minimum = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    maximum = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    center = (minimum + maximum) * 0.5
    extent = max(maximum.x - minimum.x, maximum.y - minimum.y, maximum.z - minimum.z, 1.0)
    transform = Matrix.Scale(1.0 / extent, 4) @ Matrix.Translation(-center)
    for obj in objects:
        obj.matrix_world = transform @ obj.matrix_world
    bpy.context.view_layer.update()


def set_material(color):
    mat = bpy.data.materials.new("3DViewRenderMaterial")
    mat.diffuse_color = (float(color[0]), float(color[1]), float(color[2]), 1.0)
    for obj in mesh_objects():
        obj.data.materials.clear()
        obj.data.materials.append(mat)


def look_at(camera, position, target, up):
    location = Vector(position)
    target = Vector(target)
    forward = (target - location).normalized()
    up_vec = Vector(up).normalized()
    right = forward.cross(up_vec).normalized()
    true_up = right.cross(forward).normalized()
    matrix = Matrix((
        (right.x, true_up.x, -forward.x, location.x),
        (right.y, true_up.y, -forward.y, location.y),
        (right.z, true_up.z, -forward.z, location.z),
        (0.0, 0.0, 0.0, 1.0),
    ))
    camera.matrix_world = matrix


def setup_scene(cfg):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    import_model(cfg["model_path"])
    normalize_scene()
    set_material(cfg["mesh_color"])

    scene = bpy.context.scene
    scene.render.resolution_x = int(cfg["resolution"])
    scene.render.resolution_y = int(cfg["resolution"])
    scene.render.film_transparent = False
    scene.world = bpy.data.worlds.new("3DViewRenderWorld") if scene.world is None else scene.world
    scene.world.color = tuple(cfg["background_color"])
    try:
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.display.shading.light = "STUDIO" if cfg["shading"] else "FLAT"
        scene.display.shading.color_type = "MATERIAL"
        scene.display.shading.background_type = "VIEWPORT"
        scene.display.shading.background_color = tuple(cfg["background_color"])
    except Exception:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1

    light_data = bpy.data.lights.new("KeyLight", type="SUN")
    light_data.energy = 2.0
    light = bpy.data.objects.new("KeyLight", light_data)
    light.rotation_euler = (math.radians(45.0), 0.0, math.radians(45.0))
    scene.collection.objects.link(light)

    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    return camera


def render_views(cfg):
    camera = setup_scene(cfg)
    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    for name in cfg["views"]:
        direction, up = VIEW_POSES[name]
        distance = max(float(cfg["camera_distance"]), 0.25)
        position = tuple(float(v) * distance for v in direction)
        look_at(camera, position, (0.0, 0.0, 0.0), up)
        if cfg["camera_mode"] == "orthographic":
            camera.data.type = "ORTHO"
            camera.data.ortho_scale = max(float(cfg["orthographic_scale"]), 0.05)
        else:
            camera.data.type = "PERSP"
            camera.data.angle = math.radians(float(cfg["fov_degrees"]))
        bpy.context.scene.render.filepath = os.path.join(output_dir, f"{name}.png")
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    config_path = sys.argv[-1]
    with open(config_path, "r", encoding="utf-8") as handle:
        render_views(json.load(handle))
'''
