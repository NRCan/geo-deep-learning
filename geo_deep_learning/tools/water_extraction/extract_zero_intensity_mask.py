"""Create a proper binary zero-intensity mask from an intensity raster.

Output encoding:
- 0: valid background
- 1: zero-intensity positive cue
- 255: nodata
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

_MASK_NODATA = np.uint8(255)


def _remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Drop connected components smaller than min_size."""
    if min_size <= 1:
        return mask

    labels, num_labels = ndimage.label(mask)
    if num_labels == 0:
        return mask

    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_size
    keep[0] = False
    return keep[labels]


def _fill_small_holes(mask: np.ndarray, valid_mask: np.ndarray, max_hole_size: int) -> np.ndarray:
    """Fill enclosed holes up to max_hole_size inside the valid region."""
    if max_hole_size <= 0:
        return mask

    holes = valid_mask & ~mask
    labels, num_labels = ndimage.label(holes)
    if num_labels == 0:
        return mask

    filled = mask.copy()
    height, width = mask.shape
    for label_id in range(1, num_labels + 1):
        component = labels == label_id
        size = int(component.sum())
        if size == 0 or size > max_hole_size:
            continue

        touches_boundary = (
            component[0, :].any()
            or component[-1, :].any()
            or component[:, 0].any()
            or component[:, -1].any()
        )
        if not touches_boundary:
            filled[component] = True

    return filled


def build_zero_intensity_mask(
    intensity_path: Path,
    *,
    min_component_size: int = 1,
    max_hole_size: int = 0,
) -> tuple[np.ndarray, dict]:
    """Build the binary mask and output profile."""
    with rasterio.open(intensity_path) as src:
        intensity = src.read(1)
        nodata = src.nodata
        valid_mask = np.isfinite(intensity)
        if nodata is not None:
            valid_mask &= intensity != nodata

        zero_mask = valid_mask & (intensity == 0)
        zero_mask = _remove_small_components(zero_mask, min_component_size)
        zero_mask = _fill_small_holes(zero_mask, valid_mask, max_hole_size)

        output = np.full(intensity.shape, _MASK_NODATA, dtype=np.uint8)
        output[valid_mask] = 0
        output[zero_mask] = 1

        profile = src.profile.copy()
        profile.update(dtype="uint8", nodata=int(_MASK_NODATA), count=1, compress="lzw")

    return output, profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a binary zero-intensity mask from an intensity raster",
    )
    parser.add_argument("--input-raster", type=Path, required=True)
    parser.add_argument("--output-raster", type=Path, required=True)
    parser.add_argument("--min-component-size", type=int, default=1)
    parser.add_argument("--max-hole-size", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.output_raster.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output_raster}. Use --overwrite to replace it.",
        )

    output, profile = build_zero_intensity_mask(
        args.input_raster,
        min_component_size=args.min_component_size,
        max_hole_size=args.max_hole_size,
    )

    args.output_raster.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(args.output_raster, "w", **profile) as dst:
        dst.write(output, 1)
        dst.set_band_description(1, "zero_intensity")
        dst.update_tags(1, NODATA_VALUE=str(int(_MASK_NODATA)))


if __name__ == "__main__":
    main()
