# VGGT Serve

`vggt-serve` is now structured as a backend-extensible reconstruction service. VGGT is the only implemented backend in this branch, and additional models can be added as isolated adapters plus their own Dockerfiles.

## Quick Start With Docker

This is the easiest way to run the service.

### Requirements

- Docker
- Docker Compose
- Tailscale installed, connected, and joined to the same tailnet as the client
- NVIDIA Container Toolkit if you want GPU inference

### 1. Clone and initialize the submodule

```bash
git clone <your-repo-url>
cd vggt-server
git submodule update --init --recursive
```

### 2. Start the server

Choose a port. The same port is used inside the container and on the host.

```bash
scripts/docker_compose.sh up --backend vggt --port 9000 -d
```

Useful variants:

```bash
scripts/docker_compose.sh logs --backend vggt --port 9000
scripts/docker_compose.sh down
scripts/docker_compose.sh up --backend vggt --port 9000 --cpu
RECON_SERVE_MAX_IMAGES=32 \
RECON_SERVE_VGGT_DEFAULT_DEPTH_CONF_THRESHOLD=1.0 \
scripts/docker_compose.sh up --backend vggt --port 9000 -d
```

The wrapper builds `docker/Dockerfile.common` first, then the selected backend-specific `docker/Dockerfile.<backend>`.

Notes:

- The wrapper detects the server's Tailscale IPv4 address and exposes the service
  only on that address. It refuses to start if Tailscale is unavailable.
- `--bind-address 127.0.0.1` is an explicit escape hatch for local-only
  development. Do not use `0.0.0.0` unless public exposure is intentional and
  protected separately.
- The first run can take a while because Docker builds the image.
- The first model load can also take time because VGGT weights may be downloaded.
- `--cpu` is supported, but VGGT is intended for GPU use.

### 3. Check the server

```bash
SERVER_TAILSCALE_IP="$(tailscale ip -4)"
curl "http://${SERVER_TAILSCALE_IP}:9000/healthz"
curl "http://${SERVER_TAILSCALE_IP}:9000/readyz"
```

`/readyz` returns `200` when the model is loaded and ready.

### 4. Send a test request

Using the bundled Python client:

```bash
python scripts/client_example.py \
  vggt/examples/kitchen/images/00.png \
  vggt/examples/kitchen/images/01.png \
  --base-url http://100.x.y.z:9000 \
  --scene-id kitchen-demo \
  --download-dir ./client_outputs

# For sparse or difficult sequences, lower the confidence cutoff if the
# returned point_cloud.ply is empty.
python scripts/client_example.py \
  /path/to/image_01.png \
  /path/to/image_02.png \
  --base-url http://100.x.y.z:9000 \
  --depth-conf-threshold 1.0 \
  --download-dir ./client_outputs
```

Or with `curl`:

```bash
curl -X POST http://100.x.y.z:9000/v1/reconstructions \
  -F "scene_id=kitchen-demo" \
  -F "depth_conf_threshold=5.0" \
  -F "images=@vggt/examples/kitchen/images/00.png" \
  -F "images=@vggt/examples/kitchen/images/01.png"
```

## Remote Server Usage Over Tailscale

If the Docker container runs on a remote server, install Tailscale on both the
server and client and join them to the same tailnet.

### 1. Start the server on the remote machine

On the server:

```bash
scripts/docker_compose.sh up --backend vggt --port 8080 -d
```

### 2. Find the server's Tailscale address

On the server:

```bash
tailscale ip -4
```

### 3. Run the client locally

On your local machine:

```bash
python scripts/client_example.py \
  /path/to/image_01.png \
  /path/to/image_02.png \
  --base-url http://100.x.y.z:8080 \
  --download-dir ./client_outputs
```

Replace `100.x.y.z` with the address printed on the server. A MagicDNS hostname
can be used instead when MagicDNS is enabled.

In this setup:

- your images stay on your local machine,
- inference runs on the remote server,
- the service is reachable only through the server's Tailscale interface,
- downloaded artifacts are saved on your local machine.

## Local Conda Run

If you do not want Docker, you can run the service directly.

```bash
conda env create -f environment.yml
conda activate recon-serve-py312
python scripts/check_env.py
uvicorn vggt_serve.app:app --host "$(tailscale ip -4)" --port 8000
```

## Main Endpoints

- `GET /healthz`
- `GET /readyz`
- `POST /v1/reconstructions`
- `GET /v1/artifacts/{request_id}/{name}`

## Server Outputs

Each successful request is stored on the server under:

```text
data/runs/<request_id>/
```

Typical files:

- `request.json`
- `result.json`
- `depth.npz`
- `point_cloud.ply`

## Docker Files

- [docker/Dockerfile.common](/mnt/Backup2nd/Research/vggt-serve/docker/Dockerfile.common): shared runtime base
- [docker/Dockerfile.vggt](/mnt/Backup2nd/Research/vggt-serve/docker/Dockerfile.vggt): VGGT-specific layer
- [docker/Dockerfile.map-anything](/mnt/Backup2nd/Research/vggt-serve/docker/Dockerfile.map-anything): placeholder image
- [docker/Dockerfile.pi3](/mnt/Backup2nd/Research/vggt-serve/docker/Dockerfile.pi3): placeholder image
- [docker/Dockerfile.depth-anything3](/mnt/Backup2nd/Research/vggt-serve/docker/Dockerfile.depth-anything3): placeholder image
- [docker-compose.yml](/mnt/Backup2nd/Research/vggt-serve/docker-compose.yml): base compose config
- [docker-compose.gpu.yml](/mnt/Backup2nd/Research/vggt-serve/docker-compose.gpu.yml): GPU overlay
- [scripts/docker_compose.sh](/mnt/Backup2nd/Research/vggt-serve/scripts/docker_compose.sh): compose wrapper with `--backend`
