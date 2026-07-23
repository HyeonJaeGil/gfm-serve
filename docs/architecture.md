# Architecture

GFM Serve is a model-neutral core plus independently packaged vertical model
services:

```text
packages/gfm-serve-core/       HTTP, contracts, lifecycle, storage
services/vggt/                 adapter, config, upstream, tests, image, docs
services/depth-anything-3/     adapter, config, upstream, tests, image, docs
```

The core discovers factories from the `gfm_serve.backends` entry-point group.
A production image installs the core and exactly one service package. It never
imports every supported framework, and a request cannot swap the loaded model.

## Adding a backend

A new `services/<model>/` directory owns:

1. a package implementing `ReconstructionBackend`;
2. typed request options and deployment settings;
3. an instance-level `BackendDescriptor`;
4. a pinned `upstream/` submodule;
5. adapter/conformance tests, a Dockerfile, a smoke check, and a README.

Register the factory without editing core:

```toml
[project.entry-points."gfm_serve.backends"]
my-model = "gfm_backend_my_model:create_backend"
```

Adapters validate semantic camera/option combinations, convert canonical
OpenCV inputs to upstream conventions, and normalize outputs back to common
`ViewResult` and artifact vocabulary. Backend-native outputs use namespaced
kinds. No model-specific field or conditional belongs in the central route.

Binary writes must remain beneath `BackendRunRequest.run_dir`. Descriptors must
match accepted inputs and variant-dependent outputs. A backend that fails to
load reports its error through readiness.
