from .backend import VGGTBackend, VGGTBackendOptions


def create_backend(settings):
    return VGGTBackend(settings)


__all__ = ["VGGTBackend", "VGGTBackendOptions", "create_backend"]
