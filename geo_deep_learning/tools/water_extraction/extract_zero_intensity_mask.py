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
from rasterio.windows import Window

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


class _UnionFind:
    """Small union-find helper for tiled connected components."""

    def __init__(self) -> None:
        self.parent = [0]
        self.size = [0]

    def make(self, component_size: int) -> int:
        idx = len(self.parent)
        self.parent.append(idx)
        self.size.append(component_size)
        return idx

    def find(self, idx: int) -> int:
        parent = self.parent[idx]
        while parent != idx:
            grandparent = self.parent[parent]
            self.parent[idx] = grandparent
            idx = parent
            parent = grandparent
        return idx

    def union(self, a: int, b: int) -> int:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return root_a
        if self.size[root_a] < self.size[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        self.size[root_a] += self.size[root_b]
        self.size[root_b] = 0
        return root_a

    def component_size(self, idx: int) -> int:
        return self.size[self.find(idx)]


def _iter_tile_windows(width: int, height: int, tile_size: int):
    """Yield deterministic tile windows covering the raster."""
    for row_off in range(0, height, tile_size):
        win_h = min(tile_size, height - row_off)
        for col_off in range(0, width, tile_size):
            win_w = min(tile_size, width - col_off)
            yield (row_off, col_off), Window(col_off, row_off, win_w, win_h)


def _read_zero_tile(src, window: Window) -> tuple[np.ndarray, np.ndarray]:
    """Read one tile and return valid / zero masks."""
    intensity = src.read(1, window=window)
    nodata = src.nodata
    valid_mask = np.isfinite(intensity)
    if nodata is not None:
        valid_mask &= intensity != nodata
    zero_mask = valid_mask & (intensity == 0)
    return valid_mask, zero_mask


def _build_component_index_streaming(
    intensity_path: Path,
    *,
    tile_size: int = 2048,
) -> tuple[_UnionFind, dict[tuple[int, int], np.ndarray], dict]:
    """
    First pass: build connected-component unions and remember tile borders.

    This keeps memory bounded by tile size instead of full raster size.
    """
    union_find = _UnionFind()
    right_edges: dict[tuple[int, int], np.ndarray] = {}
    bottom_edges: dict[tuple[int, int], np.ndarray] = {}
    tile_mappings: dict[tuple[int, int], np.ndarray] = {}

    with rasterio.open(intensity_path) as src:
        profile = src.profile.copy()
        for (row_off, col_off), window in _iter_tile_windows(src.width, src.height, tile_size):
            _, zero_mask = _read_zero_tile(src, window)
            labels, num_labels = ndimage.label(zero_mask)

            if num_labels == 0:
                right_edges[(row_off, col_off)] = np.zeros(int(window.height), dtype=np.int64)
                bottom_edges[(row_off, col_off)] = np.zeros(int(window.width), dtype=np.int64)
                tile_mappings[(row_off, col_off)] = np.zeros(1, dtype=np.int64)
                continue

            local_sizes = np.bincount(labels.ravel())
            local_to_global = np.zeros(num_labels + 1, dtype=np.int64)
            for label_id in range(1, num_labels + 1):
                local_to_global[label_id] = union_find.make(int(local_sizes[label_id]))
            tile_mappings[(row_off, col_off)] = local_to_global.copy()

            global_labels = local_to_global[labels]

            if row_off > 0:
                above = bottom_edges[(row_off - tile_size, col_off)]
                current_top = global_labels[0, :]
                overlap = (above > 0) & (current_top > 0)
                for upper_id, current_id in zip(above[overlap], current_top[overlap]):
                    union_find.union(int(upper_id), int(current_id))

            if col_off > 0:
                left = right_edges[(row_off, col_off - tile_size)]
                current_left = global_labels[:, 0]
                overlap = (left > 0) & (current_left > 0)
                for left_id, current_id in zip(left[overlap], current_left[overlap]):
                    union_find.union(int(left_id), int(current_id))

            right_edges[(row_off, col_off)] = global_labels[:, -1].copy()
            bottom_edges[(row_off, col_off)] = global_labels[-1, :].copy()

    return union_find, tile_mappings, profile


def build_zero_intensity_mask_streaming(
    intensity_path: Path,
    output_path: Path,
    *,
    min_component_size: int = 1,
    tile_size: int = 2048,
) -> dict:
    """
    Write the zero-intensity mask tile-by-tile.

    This path supports the pipeline's common case where hole filling is disabled.
    """
    union_find, tile_mappings, profile = _build_component_index_streaming(
        intensity_path,
        tile_size=tile_size,
    )
    keep_roots = {
        idx for idx in range(1, len(union_find.parent))
        if union_find.find(idx) == idx and union_find.size[idx] >= min_component_size
    }

    profile.update(dtype="uint8", nodata=int(_MASK_NODATA), count=1, compress="lzw")
    tmp_output = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_output.exists():
        tmp_output.unlink()

    with rasterio.open(intensity_path) as src, rasterio.open(tmp_output, "w", **profile) as dst:
        for (row_off, col_off), window in _iter_tile_windows(src.width, src.height, tile_size):
            valid_mask, zero_mask = _read_zero_tile(src, window)
            labels, num_labels = ndimage.label(zero_mask)

            output = np.full((int(window.height), int(window.width)), _MASK_NODATA, dtype=np.uint8)
            output[valid_mask] = 0

            if num_labels > 0:
                local_to_global = tile_mappings[(row_off, col_off)]
                keep_local = np.zeros(num_labels + 1, dtype=bool)
                for label_id in range(1, num_labels + 1):
                    keep_local[label_id] = union_find.find(int(local_to_global[label_id])) in keep_roots
                output[keep_local[labels]] = 1

            dst.write(output, 1, window=window)

        dst.set_band_description(1, "zero_intensity")
        dst.update_tags(1, NODATA_VALUE=str(int(_MASK_NODATA)))

    tmp_output.replace(output_path)
    return profile


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

    args.output_raster.parent.mkdir(parents=True, exist_ok=True)
    if args.max_hole_size <= 0:
        build_zero_intensity_mask_streaming(
            args.input_raster,
            args.output_raster,
            min_component_size=args.min_component_size,
        )
    else:
        output, profile = build_zero_intensity_mask(
            args.input_raster,
            min_component_size=args.min_component_size,
            max_hole_size=args.max_hole_size,
        )
        with rasterio.open(args.output_raster, "w", **profile) as dst:
            dst.write(output, 1)
            dst.set_band_description(1, "zero_intensity")
            dst.update_tags(1, NODATA_VALUE=str(int(_MASK_NODATA)))


if __name__ == "__main__":
    main()
