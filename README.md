# GFM Serve

GFM Serve exposes geometric foundation models through one stable reconstruction
API while keeping each model's options, dependencies, and GPU ownership
isolated. One service process loads exactly one backend.

| Backend | Service documentation | Inputs |
| --- | --- | --- |
| VGGT | [services/vggt/README.md](services/vggt/README.md) | images |
| Depth Anything 3 | [services/depth-anything-3/README.md](services/depth-anything-3/README.md) | images, optional cameras |

## Quick start

```bash
git submodule update --init --recursive
scripts/docker_compose.sh up --backend vggt --port 9000 --bind-address 127.0.0.1
# or
scripts/docker_compose.sh up --backend depth-anything-3 --port 9000 --bind-address 127.0.0.1
```

The wrapper builds the lightweight `docker/Dockerfile.common`, followed by the
selected `services/<backend>/Dockerfile`. Each production image installs only
the selected upstream model stack.

Inspect the active checkpoint and its variant-dependent capabilities:

```bash
curl http://127.0.0.1:9000/readyz
curl http://127.0.0.1:9000/v1/models/current
```

Send a manifest request and download its artifacts:

```bash
python scripts/client_example.py \
  services/vggt/upstream/examples/kitchen/images/00.png \
  services/vggt/upstream/examples/kitchen/images/01.png \
  --base-url http://127.0.0.1:9000 \
  --scene-id kitchen \
  --backend-options-json '{"depth_conf_threshold": 1.0}' \
  --download-dir ./client_outputs
```

The legacy repeated `images` transport remains available with
`--legacy`. See [docs/api.md](docs/api.md) for the transport contract and
[docs/migration-v1.md](docs/migration-v1.md) for compatibility dates.

## Development

```bash
python -m pip install -e 'packages/gfm-serve-core[test]'
python -m pip install --no-deps -e services/vggt
python -m pip install --no-deps -e services/depth-anything-3
pytest -q
```

For a local model environment, install only the upstream and service package
you intend to run. Start the app with:

```bash
GFM_SERVE_BACKEND=vggt uvicorn --factory gfm_serve.app:create_app --host 127.0.0.1 --port 8000
```

Architecture and extension rules are in
[docs/architecture.md](docs/architecture.md).
