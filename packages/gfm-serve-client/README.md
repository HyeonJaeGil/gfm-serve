# GFM Serve Python client

`gfm-serve-client` is the application-facing Python SDK for both GFM Serve
backends. It builds multipart manifests, validates inputs before upload, parses
typed responses, and downloads result artifacts.

## Install

From this repository:

```bash
python -m pip install -e packages/gfm-serve-client
```

The client is independent of PyTorch, VGGT, DA3, and the server runtime. An
application only needs `httpx`, `numpy`, and `pydantic`.

Each server process owns one backend. Connect `VGGTClient` to a VGGT service
and `DepthAnything3Client` to a DA3 service. Backend-specific clients raise
`BackendMismatchError` if discovery shows that the URL points to the wrong
service.

## VGGT

VGGT accepts images and predicts cameras, relative depth, confidence, and a
point cloud:

```python
from gfm_serve_client import VGGTClient

with VGGTClient("http://127.0.0.1:9000") as client:
    result = client.reconstruct(
        ["frames/000.png", "frames/001.png"],
        scene_id="office",
        view_ids=["left", "right"],
        depth_conf_threshold=1.0,
    )

    print(result.request_id, result.timings_ms.inference)
    for view in result.view_results:
        if view.camera is not None:
            print(view.view_id, view.camera.world_to_camera_array)

    paths = client.download_artifacts(
        result,
        "outputs/vggt",
        kinds={"depth_archive", "point_cloud"},
    )
```

`VGGTClient` has no `cameras` argument because the VGGT backend rejects
supplied camera calibration.

## Depth Anything 3

Image-only inference:

```python
from gfm_serve_client import DepthAnything3Client, DepthAnything3Options

with DepthAnything3Client("http://127.0.0.1:9001") as client:
    result = client.reconstruct(
        ["frames/000.png", "frames/001.png"],
        options=DepthAnything3Options(
            process_res=756,
            ref_view_strategy="middle",
        ),
    )
    arrays = client.load_depth_archive(result)
    print(arrays.keys(), arrays["depth"].shape)
```

Pose-conditioned inference accepts one `CameraParameters` object per image:

```python
import numpy as np

from gfm_serve_client import (
    CameraParameters,
    DepthAnything3Client,
    DepthAnything3Options,
)

intrinsics = [
    np.array([[800, 0, 640], [0, 800, 360], [0, 0, 1]]),
    np.array([[800, 0, 640], [0, 800, 360], [0, 0, 1]]),
]
camera_to_world = [np.eye(4), np.eye(4)]
camera_to_world[1][0, 3] = 0.1

cameras = [
    CameraParameters.from_camera_to_world(
        intrinsics=K,
        camera_to_world=c2w,
    )
    for K, c2w in zip(intrinsics, camera_to_world, strict=True)
]

with DepthAnything3Client("http://127.0.0.1:9001") as client:
    result = client.reconstruct(
        ["frames/000.png", "frames/001.png"],
        cameras=cameras,
        options=DepthAnything3Options(align_to_input_ext_scale=True),
    )
```

Intrinsics must be pixel-space `3×3` matrices for the original image
dimensions. Extrinsics use OpenCV homogeneous `4×4` world-to-camera matrices:

```text
X_camera = R @ X_world + t
world_to_camera = [[R, t], [0, 0, 0, 1]]
```

Use `CameraParameters.from_camera_to_world(...)` when the application stores
camera-to-world poses. The SDK validates shape, finiteness, the homogeneous
last row, and invertibility. Pose-conditioned requests require cameras for
every image; mono and metric-only DA3 checkpoints do not accept cameras.

## Results and artifacts

`reconstruct()` returns a `ReconstructionResult`, not a raw dictionary. Useful
fields include `request_id`, `model`, `timings_ms`, `warnings`, `view_results`,
`artifacts`, and `produced_outputs`. A returned camera's `source` is
`predicted`, `provided`, or `aligned`.

Select and download artifacts:

```python
artifact = result.artifact(kind="point_cloud")
path = client.download_artifact(artifact, "outputs/cloud.ply")

paths = client.download_artifacts(
    result,
    "outputs/all",
    overwrite=False,
)
```

`load_depth_archive(result)` downloads the `depth_archive` NPZ in memory and
returns copied NumPy arrays. For large results, prefer `download_artifact()` and
open the file with `numpy.load()` yourself.

## Discovery, readiness, and errors

```python
from gfm_serve_client import GFMServeAPIError, VGGTClient

with VGGTClient("http://127.0.0.1:9000") as client:
    assert client.health()
    readiness = client.ready()
    descriptor = client.model_descriptor()

    try:
        result = client.reconstruct(["frame.png"])
    except GFMServeAPIError as exc:
        print(exc.status_code, exc.code, exc.message, exc.request_id)
```

Use the generic `GFMServeClient` only when an application intentionally chooses
the backend at runtime. All clients are context managers. Calling `close()`
explicitly is equivalent when a context manager is inconvenient.
