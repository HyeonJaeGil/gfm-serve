# v1 migration to GFM Serve

The distribution and preferred core import changed from `vggt-serve` /
`vggt_serve` to `gfm-serve` / `gfm_serve`. The service name and container
prefix are now `gfm-serve`. The `vggt_serve` namespace forwards historical
imports and emits `DeprecationWarning`.

The reconstruction path remains `POST /v1/reconstructions`. Repeated `images`
fields, `vggt_serve` imports, `RECON_SERVE_*`, and `VGGT_SERVE_*` environment
names are supported through **2026-12-31**. They are scheduled for removal in
the next major API release after that date. New deployments should use manifest
requests and `GFM_SERVE_*`.

Legacy example:

```bash
curl -X POST http://127.0.0.1:8000/v1/reconstructions \
  -F images=@one.png \
  -F images=@two.png \
  -F scene_id=legacy-scene \
  -F depth_conf_threshold=1.0
```

`depth_conf_threshold` is translated into VGGT options and produces a
deprecation header and result warning. Other backends reject it. The preferred
equivalent is `manifest.options.depth_conf_threshold`.

Update image paths from top-level `vggt/` to
`services/vggt/upstream/`, backend Dockerfiles to
`services/<backend>/Dockerfile`, and the server entry point to
`gfm_serve.app:create_app` with Uvicorn's `--factory` flag.
