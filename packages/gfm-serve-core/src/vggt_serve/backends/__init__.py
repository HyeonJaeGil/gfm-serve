from gfm_serve.backends import *  # noqa: F403

try:
    from gfm_backend_vggt import VGGTBackend, VGGTBackendOptions
except ImportError:
    pass
