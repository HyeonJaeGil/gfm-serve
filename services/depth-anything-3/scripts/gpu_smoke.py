from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from depth_anything_3.api import DepthAnything3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Opt-in DA3 GPU inference smoke test.")
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument(
        "--model-id",
        default=os.getenv(
            "GFM_SERVE_DEPTH_ANYTHING_3_MODEL_ID",
            "depth-anything/DA3NESTED-GIANT-LARGE",
        ),
    )
    parser.add_argument(
        "--revision",
        default=os.getenv("GFM_SERVE_DEPTH_ANYTHING_3_MODEL_REVISION", "main"),
    )
    parser.add_argument("--process-res", type=int, default=504)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("DA3 GPU smoke requires a CUDA device.")
    image_paths = [path.expanduser().resolve() for path in args.images]
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing input images: {', '.join(missing)}")

    model = DepthAnything3.from_pretrained(
        args.model_id,
        revision=args.revision,
    ).to("cuda")
    model.eval()
    prediction = model.inference(
        image=[str(path) for path in image_paths],
        process_res=args.process_res,
    )

    depth = np.asarray(prediction.depth)
    if depth.ndim != 3 or depth.shape[0] != len(image_paths):
        raise RuntimeError(
            f"Expected depth shape (N,H,W) for {len(image_paths)} views, got {depth.shape}."
        )
    if not np.isfinite(depth).all() or not np.any(depth > 0):
        raise RuntimeError("DA3 produced non-finite or entirely non-positive depth.")
    if prediction.conf is not None:
        confidence = np.asarray(prediction.conf)
        if confidence.shape != depth.shape or not np.isfinite(confidence).all():
            raise RuntimeError("DA3 confidence shape/finiteness check failed.")
    if prediction.extrinsics is not None and len(prediction.extrinsics) != len(image_paths):
        raise RuntimeError("DA3 extrinsic view count does not match the input.")
    if prediction.intrinsics is not None and len(prediction.intrinsics) != len(image_paths):
        raise RuntimeError("DA3 intrinsic view count does not match the input.")

    print(
        f"DA3 GPU smoke passed: model={args.model_id} views={len(image_paths)} "
        f"depth_shape={depth.shape}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
