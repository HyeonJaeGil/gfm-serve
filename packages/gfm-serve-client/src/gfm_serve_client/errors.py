from __future__ import annotations

from typing import Any


class GFMServeError(Exception):
    """Base class for client-side and service-side GFM Serve failures."""


class InvalidRequestError(GFMServeError, ValueError):
    """Raised before transport when client input is inconsistent."""


class BackendMismatchError(GFMServeError):
    """Raised when a backend-specific client is connected to the wrong service."""

    def __init__(self, *, expected: str, actual: str) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(f"Expected backend '{expected}', but the service runs '{actual}'.")


class GFMServeAPIError(GFMServeError):
    """A structured error returned by GFM Serve."""

    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        request_id: str | None = None,
        response_payload: dict[str, Any] | None = None,
    ) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        self.request_id = request_id
        self.response_payload = response_payload
        request_suffix = f" (request_id={request_id})" if request_id else ""
        super().__init__(f"{code}: {message}{request_suffix}")
