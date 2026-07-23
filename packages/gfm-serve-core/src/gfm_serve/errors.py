from __future__ import annotations


class ApiError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class ValidationApiError(ApiError):
    def __init__(self, message: str) -> None:
        super().__init__(400, "validation_error", message)


class UnsupportedMediaApiError(ApiError):
    def __init__(self, message: str) -> None:
        super().__init__(415, "unsupported_media", message)


class ServiceBusyApiError(ApiError):
    def __init__(self, message: str = "The reconstruction service is already processing another request.") -> None:
        super().__init__(503, "service_busy", message)


class ExecutionFailedApiError(ApiError):
    def __init__(self, message: str = "The reconstruction workflow failed during execution.") -> None:
        super().__init__(500, "execution_failed", message)


class ServiceUnavailableApiError(ApiError):
    def __init__(self, message: str = "The reconstruction service is not ready.") -> None:
        super().__init__(503, "service_unavailable", message)


class ArtifactNotFoundApiError(ApiError):
    def __init__(self, message: str = "Requested artifact was not found.") -> None:
        super().__init__(404, "artifact_not_found", message)

