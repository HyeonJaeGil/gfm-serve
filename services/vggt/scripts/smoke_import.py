from importlib.metadata import entry_points

from gfm_backend_vggt import VGGTBackend
from vggt.models.vggt import VGGT


assert VGGT is not None
assert VGGTBackend.backend_id == "vggt"
assert any(
    entry_point.name == "vggt"
    for entry_point in entry_points(group="gfm_serve.backends")
)
