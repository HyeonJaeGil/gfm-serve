from gfm_serve.app import create_app
from gfm_serve.config import Settings


# Retain the historical ASGI target for deployments that install only VGGT.
app = create_app(settings=Settings(backend="vggt"))

__all__ = ["app", "create_app"]
