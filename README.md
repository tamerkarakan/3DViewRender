# 3D View Render

ComfyUI custom node for rendering selected six-side views from a `MESH` input.

Compatibility target: ComfyUI `v0.22.0` V3 custom-node API. The package exports a V3 `comfy_entrypoint()` when `comfy_api.latest` is available, and exposes legacy `NODE_CLASS_MAPPINGS` only as a fallback for older installs.

## Node

`3D View Render: Six Sides`

Inputs:

- `model`: ComfyUI `MESH`, `TRIMESH`, `MESHWITHVOXEL`, `FILE_3D*`, or a 3D file path string
- `resolution`: square output image size
- `camera_mode`: `orthographic` or `perspective`
- `front`, `back`, `left`, `right`, `top`, `bottom`: side selection toggles
- advanced controls: `background_color`, `mesh_color`, `fov_degrees`, `orthographic_scale`, `camera_distance`, `shading`

Outputs:

- `images`: ComfyUI `IMAGE` batch in selected side order for each mesh batch item
- `view_names`: newline-separated labels such as `0:front`

## Tests

The core renderer tests do not require ComfyUI or torch.

```powershell
py -3.14 -m unittest discover -s tests -v
```
