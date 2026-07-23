# Client guide

GFM Serve can be called through raw HTTP or the typed Python SDK. This page is
the common entry point; detailed documentation remains next to the contract or
package it describes.

| Interface | Best for | Documentation | Executable example |
| --- | --- | --- | --- |
| `curl` / raw multipart HTTP | debugging, shell scripts, non-Python clients | [HTTP API](api.md) | commands in `docs/api.md` |
| low-level Python `httpx` | custom transport behavior and legacy requests | [HTTP API](api.md) | [`scripts/client_example.py`](../scripts/client_example.py) |
| typed Python SDK | application code | [SDK reference](../packages/gfm-serve-client/README.md) | [`examples/vggt_client.py`](../examples/vggt_client.py), [`examples/depth_anything_3_client.py`](../examples/depth_anything_3_client.py) |

## Why the detailed documents live in different places

`docs/api.md` defines the language-independent HTTP wire contract implemented
by the server. A `curl`, Rust, JavaScript, or custom Python client all depend on
that same contract.

`packages/gfm-serve-client/README.md` documents the installable
`gfm-serve-client` Python distribution. Keeping its API reference beside its
`pyproject.toml` and source means the package can be versioned, built, or
published with its own documentation intact.

This guide links both views so users do not need to know the repository layout
before choosing an interface.

## Recommended choice

Use `VGGTClient` or `DepthAnything3Client` in Python applications. They validate
the connected backend, camera matrices, options, structured errors, and typed
results. Use raw HTTP when integrating another language or inspecting the exact
multipart request.

Both VGGT and DA3 image-only curl requests, plus the complete pose-conditioned
DA3 curl request, are in [the HTTP API documentation](api.md). Equivalent typed
Python examples are in the SDK reference and `examples/`.
