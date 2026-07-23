# GFM Serve: multi-model repository and API migration plan

Status: proposed

Initial models: VGGT and Depth Anything 3 (DA3)

Recommended repository name: `gfm-serve`

Recommended Python package: `gfm_serve`

## 1. Goal

Turn the current VGGT-oriented service into a service for geometric foundation
models (GFMs), without reducing every model to the lowest common denominator.
The service should:

- keep upload, validation, persistence, error handling, and artifact delivery
  model-independent;
- let each backend declare and validate its own optional inputs and inference
  options;
- give common geometric concepts (views, cameras, depth, point clouds) stable
  service-level representations;
- isolate upstream model source code and heavyweight dependencies;
- support adding another model without editing the central API route for every
  new field;
- preserve the existing VGGT API during a documented migration window.

This plan keeps one loaded backend per service process/container. Selecting a
different backend per request would make GPU memory ownership, readiness, and
dependency isolation substantially harder. Multiple model services can still be
run side by side on different ports.

## 2. Findings in the current repository

Useful foundations already exist:

- `ReconstructionBackend`, a registry, and backend-specific option models;
- a common Docker image plus backend-specific Dockerfiles;
- common artifact storage and a backend-neutral `RECON_SERVE_BACKEND` setting;
- capability reporting through `/readyz`.

The remaining coupling is structural:

- the application package and distribution are named `vggt_serve` /
  `vggt-serve`;
- the VGGT upstream submodule occupies the root-level `vggt/` path;
- settings contain VGGT fields directly on the root settings object;
- the multipart endpoint only accepts a positional `images[]` list;
- `depth_conf_threshold` is a VGGT-specific top-level form field;
- `BackendRunRequest` contains images and options, but no typed per-view
  metadata;
- `ReconstructionResponse.camera_results` assumes every returned camera has
  both intrinsics and extrinsics;
- the common Docker build copies the VGGT-named application package.

DA3 makes these limitations visible. Its official Python API accepts optional
world-to-camera extrinsics shaped `(N, 4, 4)` and intrinsics shaped `(N, 3, 3)`
alongside the image list. Its prediction may contain depth, confidence,
extrinsics, intrinsics, processed images, and auxiliary outputs. Some options
and outputs are model-variant-dependent, so "`depth-anything3` supports X" is
not precise enough without describing the selected checkpoint/variant.

## 3. Naming decision

Use **GFM Serve** as the product name and `gfm-serve` as the repository and
distribution name. It is short, includes the intended model family, and does
not imply that the service only estimates depth or only reconstructs point
clouds.

Migration mapping:

| Current | Target |
| --- | --- |
| repository `vggt-serve` | `gfm-serve` |
| distribution `vggt-serve` | `gfm-serve` |
| package `vggt_serve` | `gfm_serve` |
| service label `VGGT Serve` | `GFM Serve` |
| generic env prefix `RECON_SERVE_` | `GFM_SERVE_` |
| model env prefix `RECON_SERVE_VGGT_` | `GFM_SERVE_VGGT_` |

Keep the current `RECON_SERVE_*` and legacy `VGGT_SERVE_*` aliases for one
deprecation cycle. Do not rename the HTTP path `/v1/reconstructions` merely for
branding: reconstruction remains the shared operation and retaining it avoids
needless client breakage.

When the remote GitHub repository is renamed, existing GitHub clone URLs should
redirect, but CI variables, image names, documentation links, deployment paths,
and local remotes must still be audited explicitly.

## 4. Target repository layout

```text
gfm-serve/
├── packages/
│   └── gfm-serve-core/
│       ├── pyproject.toml
│       ├── src/gfm_serve/
│       │   ├── api/
│       │   │   ├── routes.py
│       │   │   ├── request_parser.py
│       │   │   └── schemas.py
│       │   ├── contracts/
│       │   │   ├── backend.py
│       │   │   ├── inputs.py
│       │   │   └── outputs.py
│       │   ├── runtime/
│       │   │   ├── registry.py
│       │   │   └── config.py
│       │   ├── artifacts.py
│       │   ├── errors.py
│       │   ├── storage.py
│       │   └── app.py
│       └── tests/
├── services/
│   ├── vggt/
│   │   ├── pyproject.toml
│   │   ├── src/gfm_backend_vggt/
│   │   │   ├── backend.py
│   │   │   ├── config.py
│   │   │   ├── inputs.py
│   │   │   └── outputs.py
│   │   ├── upstream/          # VGGT git submodule
│   │   ├── tests/
│   │   ├── Dockerfile
│   │   └── README.md
│   └── depth-anything-3/
│       ├── pyproject.toml
│       ├── src/gfm_backend_depth_anything_3/
│       │   ├── backend.py
│       │   ├── config.py
│       │   ├── inputs.py
│       │   └── outputs.py
│       ├── upstream/          # DA3 git submodule
│       ├── tests/
│       ├── Dockerfile
│       └── README.md
├── docker/
│   └── Dockerfile.common
├── docs/
│   ├── architecture.md
│   ├── api.md
│   └── migration-v1.md
├── scripts/
├── README.md
└── pyproject.toml
```

This is a vertical-slice layout. Each `services/<model>/` directory owns the
adapter, configuration, dependency declaration, upstream submodule, tests,
Dockerfile, and model-specific documentation required to operate that model.
Adding or removing a model therefore has a local, reviewable change boundary
instead of scattering changes across `third_party/`, `docker/`, `docs/models/`,
and the core package.

`packages/gfm-serve-core/` owns only the HTTP transport, stable contracts,
storage, runtime lifecycle, and artifact vocabulary shared by all model
services. A service package depends on the core package and exposes its backend
factory through a Python entry point such as:

```toml
[project.entry-points."gfm_serve.backends"]
vggt = "gfm_backend_vggt:create_backend"
```

The core runtime discovers the one installed backend package at startup. A
production model image installs `gfm-serve-core` plus exactly one service
package and its upstream dependency. It must fail startup if zero or multiple
backend entry points are installed unless an explicit backend selection is
provided.

The `upstream/` child is a pinned external source tree, not a location for
service adapter code. Keeping it beside its owner still preserves that
distinction while making model updates atomic with adapter and image changes.

The DA3 submodule should use the requested upstream
`git@github.com:ByteDance-Seed/Depth-Anything-3.git` URL if all build hosts have
SSH credentials. If unauthenticated CI or external contributors must initialize
submodules, prefer the HTTPS URL or configure a CI-only URL rewrite.

## 5. Scalable input contract

### 5.1 Separate data, inference options, and deployment configuration

These are different concerns and must not share one untyped dictionary:

1. **Scene data**: images and metadata that describe observations, such as
   intrinsics and extrinsics.
2. **Inference options**: request-scoped behavior such as a confidence threshold
   or DA3 reference-view strategy.
3. **Backend configuration**: startup-scoped model ID, checkpoint, device, and
   memory-related settings.

Define service-level types for concepts shared by more than one GFM:

```python
class CameraInput(BaseModel):
    convention: Literal["opencv"]
    world_to_camera: Matrix4x4 | None = None
    intrinsics: Matrix3x3 | None = None

class ViewInput(BaseModel):
    view_id: str
    upload_key: str
    camera: CameraInput | None = None

class SceneInput(BaseModel):
    scene_id: str | None = None
    views: list[ViewInput]
```

Use one explicit canonical convention at the HTTP boundary:

- extrinsics are homogeneous `4x4` world-to-camera matrices;
- intrinsics are pixel-space `3x3` matrices for the uploaded image's original
  width and height;
- matrices contain finite numbers;
- the number of views, uploads, camera entries, and output records must match;
- the backend adapter owns conversion to and from upstream conventions.

Do not infer conventions from matrix shape. Future convention support should add
an explicit enum and conversion routine.

### 5.2 Multipart transport

Retain multipart uploads, but add a JSON `manifest` part that associates stable
view IDs with file parts. Example:

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
    {
      "view_id": "cam-001",
      "upload_key": "image_001"
    }
  ],
  "options": {
    "ref_view_strategy": "middle",
    "align_to_input_ext_scale": true
  }
}
```

Files use matching multipart field names (`image_000`, `image_001`). This avoids
fragile reliance on repeated-field ordering and lets metadata be optional per
view.

For v1 compatibility, continue accepting repeated `images` fields when
`manifest` is absent and synthesize view IDs from their order. Reject a request
that mixes the legacy and manifest forms. Remove the top-level
`depth_conf_threshold` only in a future major API version; until then translate
it into VGGT options and return a deprecation signal.

### 5.3 Backend-owned validation

Replace `validate_options()` with validation of a complete backend input:

```python
class ReconstructionBackend(ABC):
    input_model: type[BackendInput]
    options_model: type[BaseModel]
    descriptor: BackendDescriptor

    def validate_request(
        self,
        scene: SceneInput,
        options: dict[str, object],
    ) -> ValidatedBackendRequest: ...
```

The common parser validates file integrity and common camera types first. The
selected backend then enforces semantic combinations:

- VGGT initially rejects supplied cameras as unsupported input;
- DA3 accepts images alone, or pose-conditioned input when the selected variant
  supports it;
- DA3 requires intrinsics/extrinsics combinations according to its upstream
  contract and validates view counts and shapes;
- variant-only flags such as Gaussian inference are rejected unless the loaded
  model descriptor advertises them.

This provides strict typing without adding DA3 conditionals to the FastAPI
route. If a future model needs non-camera metadata, add a typed
backend-specific `model_input` object to its input model. Promote a field into
the common `ViewInput` only after it has a stable cross-model meaning.

### 5.4 Capability discovery

Replace a flat output-only tuple with a structured descriptor, exposed by
`GET /v1/models/current` and included in readiness:

```json
{
  "backend": "depth-anything-3",
  "model_id": "depth-anything/DA3NESTED-GIANT-LARGE",
  "inputs": {
    "images": {"required": true},
    "camera.intrinsics": {"required": false},
    "camera.world_to_camera": {"required": false}
  },
  "outputs": ["depth", "depth_confidence", "camera_poses", "point_cloud"],
  "options_schema": { "...": "JSON Schema generated by Pydantic" }
}
```

Descriptors are instance-level because the selected checkpoint can change
capabilities. Keep `/readyz` concise and link or point clients to the full
descriptor.

## 6. Output and artifact contract

Make camera fields optional and key all per-view outputs by `view_id`, not only
by filename:

```python
class ViewResult(BaseModel):
    view_id: str
    filename: str
    original_size: ImageSize
    camera: CameraResult | None = None

class CameraResult(BaseModel):
    convention: Literal["opencv"]
    world_to_camera: Matrix4x4 | None = None
    intrinsics: Matrix3x3 | None = None
    source: Literal["predicted", "provided", "aligned"]
```

Keep large tensors out of the JSON response. Store depth, confidence, point
clouds, and future Gaussian data as versioned artifacts with a manifest that
describes dtype, shape, units/scale semantics, coordinate convention, and
producing backend/model. Artifact kinds should be stable service vocabulary;
backend-native exports can use namespaced kinds such as
`depth-anything-3/gaussian-splats`.

The result manifest must record:

- service and result schema versions;
- backend ID and exact model/checkpoint revision;
- normalized validated request (excluding binary data);
- input and output coordinate conventions;
- produced outputs and artifacts;
- timing and warnings, including ignored or aligned camera data.

## 7. Configuration and dependency isolation

- Make root settings genuinely common (`service`, limits, storage, backend).
- Move `VGGTBackendSettings` and new `DA3BackendSettings` into their respective
  `services/<model>/src/` packages.
- Discover backend factories through the `gfm_serve.backends` package entry
  point group, rather than importing every future model from the core package
  or adding its fields to root `Settings`.
- Adopt `GFM_SERVE_BACKEND`, `GFM_SERVE_VGGT_*`, and
  `GFM_SERVE_DEPTH_ANYTHING_3_*`, with temporary aliases as described above.
- Pin each upstream submodule commit and record the compatible model revision.
- Build a common lightweight core wheel and install only one service package and
  upstream model stack in each model image.
- Let each `services/<model>/Dockerfile` copy its local `upstream/`, install the
  pinned model requirements, and install its service package plus the common
  core wheel. Do not install both VGGT and DA3 in the common image.
- Add backend-specific environment checks/smoke tests; a common check cannot
  prove that CUDA extensions and model imports work for every image.

Use separate production containers per backend. A future multi-GPU router can
sit in front of them; it should not be coupled to the in-process adapter
registry.

## 8. Documentation ownership

The root `README.md` should contain only shared information:

- what GFM Serve is;
- supported model table and links;
- common API/quick-start;
- deployment topology;
- links to architecture and migration documentation.

Each `services/<model>/README.md` owns:

- supported upstream versions and model variants;
- accepted inputs and valid combinations;
- model-specific request options;
- outputs, conventions, scale guarantees, and limitations;
- required hardware and image build/run commands;
- checkpoint/environment variables;
- a tested request example.

`docs/api.md` owns the transport and common schema. `docs/architecture.md` owns
extension rules, including the required service package entry point. Avoid
copying upstream READMEs; link to upstream and document only service integration
behavior.

## 9. Patch sequence

Each phase below is a focused feature-level commit. Tests must pass after every
commit unless a commit is explicitly marked as a mechanical rename paired with
the immediately following fixup.

### Phase 1 — Freeze the common contract

1. Add matrix, camera, view, scene, backend descriptor, and view-result models.
2. Add pure validation/conversion tests, including non-finite matrices, wrong
   shapes, duplicate view IDs, and mismatched upload keys.
3. Refactor `BackendRunRequest` to carry the validated scene and prepared views.
4. Adapt VGGT internally without changing the existing HTTP request/response.

Acceptance: all existing API tests pass and new contract tests pass.

### Phase 2 — Add manifest requests and discovery

1. Implement multipart `manifest` parsing and stable upload-key association.
2. Preserve the legacy repeated `images` path and reject ambiguous mixed input.
3. Move VGGT's threshold into its typed options while retaining compatibility
   translation for the old form field.
4. Add `GET /v1/models/current` from the active backend descriptor.
5. Persist the normalized manifest and schema version.

Acceptance: API tests cover image-only legacy requests, manifest image-only
requests, camera metadata, invalid matrices, missing/extra file parts, unknown
options, and discovery JSON Schema.

### Phase 3 — Make the repository model-neutral

1. Rename `vggt_serve/` to `gfm_serve/`, imports, distribution metadata,
   entrypoints, container copy paths, environment defaults, log messages, and
   tests.
2. Extract shared runtime code into `packages/gfm-serve-core/` and package it as
   `gfm-serve-core`.
3. Create `services/vggt/`, move the VGGT adapter and configuration into its
   service package, and register it through the backend entry point.
4. Move the VGGT submodule from `vggt/` to `services/vggt/upstream/` with
   `.gitmodules` updated, preserving its pinned commit.
5. Move the VGGT Dockerfile, tests, and model-specific documentation into
   `services/vggt/`.
6. Introduce `GFM_SERVE_*` variables and test legacy aliases.
7. Update scripts to discover service directories and their Dockerfiles instead
   of maintaining a hard-coded model-to-asset mapping.

Acceptance: no service-owned module or generic UI string is VGGT-named; VGGT
still builds and serves the compatibility request; the common core can be
installed and tested without either upstream model.

### Phase 4 — Integrate Depth Anything 3

1. Create `services/depth-anything-3/` and add its `upstream/` submodule at a
   reviewed, pinned DA3 commit.
2. Implement its independent Python service package with `DA3BackendSettings`,
   request options, descriptor, adapter, backend entry point, and output
   normalization.
3. Pass ordered image paths and optional `(N, 4, 4)` world-to-camera /
   `(N, 3, 3)` intrinsic arrays to `DepthAnything3.inference`.
4. Normalize DA3's returned `(N, 3, 4)` extrinsics to service `4x4` matrices
   and clearly label whether cameras were predicted, provided, or aligned.
5. Produce the common versioned depth/confidence artifact and point cloud where
   the necessary geometry is available. Keep DA3-native exports namespaced.
6. Implement `services/depth-anything-3/Dockerfile`, compose variables, asset
   checks, and a backend smoke test.
7. Add `services/depth-anything-3/README.md`.

Acceptance: DA3 tests cover images-only inference, pose-conditioned inference,
partial/invalid camera input, variant-restricted options, output convention
conversion, CPU-stubbed contract execution, and a separately runnable GPU smoke
test.

### Phase 5 — Documentation, rename, and release

1. Rewrite root `README.md` as the shared GFM Serve entry point.
2. Complete `services/vggt/README.md`, `docs/api.md`,
   `docs/architecture.md`, and `docs/migration-v1.md`.
3. Update examples/client code to generate manifest requests, with a legacy
   example retained in migration docs.
4. Rename container images, Conda environment, cache/build labels, badges, and
   deployment references.
5. Rename the remote repository to `gfm-serve` only after code and documentation
   no longer assume the old directory name.
6. Publish deprecation dates for legacy package imports and environment names.

Acceptance: a clean clone can initialize both submodules, build either backend
image independently, run its documented request, and retrieve a self-describing
result manifest.

## 10. Test strategy

- **Contract tests** run without model dependencies and are parameterized over
  every registered backend descriptor.
- **Adapter unit tests** mock upstream inference only at the library boundary
  and verify exact arrays, ordering, conversions, and option forwarding.
- **API tests** use stub backends but exercise real multipart parsing,
  persistence, error mapping, and compatibility behavior.
- **Image smoke tests** import and load the configured backend in its own Docker
  image.
- **GPU integration tests** use a tiny fixture set, are opt-in, and verify shape,
  finiteness, conventions, and artifact readability rather than unstable
  numerical equality.
- **Documentation examples** are executable tests where practical.

Add a reusable backend conformance suite. A newly registered backend must prove
that its descriptor matches accepted inputs, unknown fields are rejected,
results use stable view IDs, artifacts stay inside the run directory, and
readiness failures are reported consistently.

## 11. Risks and explicit non-goals

- Camera convention and scale ambiguity is the largest correctness risk.
  Conversion must be centralized and tested with synthetic cameras before DA3
  is exposed.
- DA3 capabilities differ by variant; discovery must describe the loaded model,
  not merely the adapter.
- SSH submodules can break unattended builds without credentials.
- Moving a submodule and renaming the Python package creates a large diff; do it
  only after contract tests protect behavior.
- Independently packaged services can drift from the core contract; pin their
  supported core version and run the shared conformance suite against every
  service in CI.
- Installing both frameworks into one environment invites dependency conflicts
  and excessive image size; backend-specific images remain the boundary.

Not included in this migration:

- dynamically swapping models inside one process;
- a scheduler spanning multiple model containers;
- arbitrary upstream export flags passed through unchecked;
- uploading camera metadata as opaque files without a service schema;
- duplicating entire upstream model documentation in this repository.

## 12. Definition of done

The migration is complete when adding a third GFM requires only:

1. one new `services/<model>/` directory;
2. an independent backend package implementing the common contract and entry
   point;
3. its local pinned `upstream/` submodule;
4. typed backend input/options/configuration models and a descriptor;
5. its local Dockerfile, conformance/adapter tests, and README;

and does **not** require adding model-specific form fields, response fields, or
conditionals to the central API route, editing the core package, or placing
files in unrelated top-level model directories.

## 13. References

- [Depth Anything 3 upstream repository](https://github.com/ByteDance-Seed/Depth-Anything-3)
- [Depth Anything 3 Python API](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/API.md)
