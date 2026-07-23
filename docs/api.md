# GFM Serve API

## Discovery and health

- `GET /healthz` reports HTTP process health.
- `GET /readyz` reports the selected backend's concise readiness state.
- `GET /v1/models/current` returns the exact model/checkpoint descriptor,
  accepted common inputs, produced outputs, and the backend options JSON Schema.

## Create a reconstruction

`POST /v1/reconstructions` accepts multipart form data. The preferred transport
has one JSON `manifest` field and uniquely named image parts:

```json
{
  "scene_id": "office",
  "views": [
    {
      "view_id": "cam-000",
      "upload_key": "image_000",
      "camera": {
        "convention": "opencv",
        "world_to_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        "intrinsics": [[800, 0, 640], [0, 800, 360], [0, 0, 1]]
      }
    },
    {"view_id": "cam-001", "upload_key": "image_001"}
  ],
  "options": {"ref_view_strategy": "middle"}
}
```

```bash
curl -X POST http://127.0.0.1:8000/v1/reconstructions \
  -F 'manifest={"views":[{"view_id":"a","upload_key":"image_000"}],"options":{}}' \
  -F image_000=@image.png
```

File order is irrelevant; `upload_key` determines view order. View IDs and
upload keys must be unique. Extra, missing, duplicated, mixed legacy/manifest,
non-image, empty, corrupt, and oversized file parts are rejected.

The HTTP convention is OpenCV. Extrinsics are finite homogeneous 4×4
world-to-camera matrices. Intrinsics are finite pixel-space 3×3 matrices for
the original upload dimensions. A backend may reject otherwise valid common
camera data when its checkpoint does not support pose conditioning.

Legacy v1 requests repeat the `images` field and may use `scene_id`,
`backend_options`, and VGGT-only `depth_conf_threshold`. The threshold response
contains `Deprecation: true` and a warning.

## Results and artifacts

Per-view results are keyed by stable `view_id`. Camera matrices are optional and
carry a `source` of `predicted`, `provided`, or `aligned`. Large arrays remain
in downloadable artifacts.

`result.json` is schema version `1.0` and records the service version, exact
backend descriptor, normalized validated request (without binary data),
coordinate conventions, produced outputs, versioned artifact metadata,
timings, and warnings. `depth_archive` artifacts contain float32 original-resolution depth
and confidence arrays. Point clouds use the OpenCV-derived world frame.

Artifacts are retrieved from
`GET /v1/artifacts/{request_id}/{name}` and are confined to the request run
directory.
