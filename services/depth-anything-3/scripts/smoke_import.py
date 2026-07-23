from importlib.metadata import entry_points

from depth_anything_3.api import DepthAnything3
from gfm_backend_depth_anything_3 import DA3Backend


assert DepthAnything3 is not None
assert DA3Backend.backend_id == "depth-anything-3"
assert any(
    entry_point.name == "depth-anything-3"
    for entry_point in entry_points(group="gfm_serve.backends")
)
