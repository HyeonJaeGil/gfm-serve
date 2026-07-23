from __future__ import annotations

from pydantic import ValidationError

from .backends import ReconstructionBackend


def assert_backend_conformance(backend: ReconstructionBackend) -> None:
    """Dependency-light checks shared by every service adapter test suite."""
    descriptor = backend.descriptor
    assert descriptor.backend == backend.backend_id
    assert descriptor.model_id
    assert descriptor.inputs.get("images") is not None
    assert descriptor.inputs["images"].required is True
    assert set(backend.capabilities).issubset(set(descriptor.outputs))
    assert descriptor.options_schema == backend.options_model.model_json_schema()
    try:
        backend.validate_options({"__unknown_conformance_field__": True})
    except ValidationError:
        pass
    else:
        raise AssertionError("Backend options must reject unknown fields.")
