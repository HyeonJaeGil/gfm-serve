from __future__ import annotations

import importlib
import sys


MODULES = [
    "fastapi",
    "uvicorn",
    "pydantic_settings",
]


def main() -> int:
    failed = False

    print(f"python: {sys.version.split()[0]}")
    for module_name in MODULES:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "unknown")
            print(f"{module_name}: OK {version}")
        except Exception as exc:  # pragma: no cover - explicit smoke path
            failed = True
            print(f"{module_name}: FAIL {type(exc).__name__}: {exc}")

    for import_stmt in ("from gfm_serve.app import create_app",):
        try:
            exec(import_stmt, {})
            print(f"{import_stmt}: OK")
        except Exception as exc:  # pragma: no cover - explicit smoke path
            failed = True
            print(f"{import_stmt}: FAIL {type(exc).__name__}: {exc}")

    try:
        import torch

        print(f"torch: OK {getattr(torch, '__version__', 'unknown')}")
        print(f"cuda_available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            capability = torch.cuda.get_device_capability(0)
            print(f"cuda_device_0: {device_name} cc={capability}")
    except Exception as exc:  # pragma: no cover - explicit smoke path
        print(f"torch: SKIP {type(exc).__name__}: {exc}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
