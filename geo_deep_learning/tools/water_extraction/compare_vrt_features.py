"""Compare stacked VRT bands against source feature rasters."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import Window

_FEATURE_ALIASES = {
    "nadir": "nadir_weighted",
    "nadir_weighted_aligned": "nadir_weighted",
    "zero": "zero_intensity",
    "zero_intensity_aligned": "zero_intensity",
}

_FLOAT_TOLERANCE = 1e-6
_DEFAULT_SAMPLE_STEP = 8


def _normalize_feature_name(feature_name: str) -> str:
    """Normalize feature names used in raster descriptions and file names."""
    normalized = feature_name.strip().lower()
    return _FEATURE_ALIASES.get(normalized, normalized)


def _band_nodata(src: rasterio.io.DatasetReader, band_index: int) -> float | None:
    """Resolve per-band nodata from dataset metadata or band tags."""
    nodata_val = src.nodatavals[band_index - 1]
    if nodata_val is not None:
        return nodata_val

    band_tags = src.tags(band_index)
    band_nodata = band_tags.get("NODATA_VALUE")
    if band_nodata is None:
        return None

    try:
        return float(band_nodata)
    except (TypeError, ValueError):
        return None


def _valid_mask(data: np.ndarray, nodata_val: float | None) -> np.ndarray:
    """Build a validity mask for a single raster band."""
    mask = np.isfinite(data)
    if nodata_val is None:
        return mask
    if isinstance(nodata_val, float) and math.isnan(nodata_val):
        return mask & ~np.isnan(data)
    return mask & (data != nodata_val)


@dataclass
class BandStats:
    """Summary statistics for a single raster band."""

    path: str
    band: int
    description: str
    shape: tuple[int, int]
    nodata: float | None
    valid_pixels: int
    min: float
    max: float
    mean: float
    std: float


@dataclass
class BandComparison:
    """Comparison between one stack band and one expected source raster."""

    band: int
    description: str
    expected_feature: str
    expected_source: str | None
    exact_match: bool | None
    same_grid: bool
    max_abs_diff: float | None
    mean_diff: float | None
    std_diff: float | None
    best_matching_feature: str | None
    order_matches_description: bool


def _iter_windows(
    src: rasterio.io.DatasetReader,
    *,
    max_block_size: int = 1024,
) -> list[Window]:
    """Return windows that cover the raster using native blocks when possible."""
    if src.block_shapes:
        windows = []
        seen: set[tuple[int, int, int, int]] = set()
        for _, window in src.block_windows(1):
            key = (
                int(window.col_off),
                int(window.row_off),
                int(window.width),
                int(window.height),
            )
            if key not in seen:
                seen.add(key)
                windows.append(window)
        return windows

    windows = []
    for row_off in range(0, src.height, max_block_size):
        height = min(max_block_size, src.height - row_off)
        for col_off in range(0, src.width, max_block_size):
            width = min(max_block_size, src.width - col_off)
            windows.append(Window(col_off, row_off, width, height))
    return windows


def _compute_band_stats(path: str | Path, *, band: int = 1) -> BandStats:
    """Compute statistics for a single raster band."""
    path = str(path)
    with rasterio.open(path) as src:
        nodata_val = _band_nodata(src, band)
        description = src.descriptions[band - 1] or Path(path).stem
        valid_pixels = 0
        sum_pixels = 0.0
        sum_sq_pixels = 0.0
        min_value = None
        max_value = None

        for window in _iter_windows(src):
            data = src.read(band, window=window, out_dtype=np.float32)
            mask = _valid_mask(data, nodata_val)
            if not np.any(mask):
                continue

            valid = data[mask]
            valid_pixels += int(valid.size)
            sum_pixels += float(valid.sum(dtype=np.float64))
            sum_sq_pixels += float(np.square(valid, dtype=np.float64).sum(dtype=np.float64))
            block_min = float(valid.min())
            block_max = float(valid.max())
            min_value = block_min if min_value is None else min(min_value, block_min)
            max_value = block_max if max_value is None else max(max_value, block_max)

        if valid_pixels == 0:
            raise ValueError(
                f"No valid pixels found in {path} band {band}. "
                "Check nodata metadata and CRS alignment.",
            )

        mean = sum_pixels / valid_pixels
        variance = max((sum_sq_pixels / valid_pixels) - (mean**2), 0.0)
        return BandStats(
            path=path,
            band=band,
            description=_normalize_feature_name(description),
            shape=(src.height, src.width),
            nodata=nodata_val,
            valid_pixels=valid_pixels,
            min=float(min_value),
            max=float(max_value),
            mean=float(mean),
            std=float(math.sqrt(variance)),
        )


def _stack_band_stats(stacked_inputs: str | Path) -> list[BandStats]:
    """Compute per-band statistics for a stacked raster or VRT."""
    stacked_inputs = str(stacked_inputs)
    with rasterio.open(stacked_inputs) as src:
        band_count = src.count
    return [
        _compute_band_stats(stacked_inputs, band=band_idx)
        for band_idx in range(1, band_count + 1)
    ]


def _resolve_expected_sources(
    source_dir: str | Path | None,
    explicit_sources: dict[str, str] | None,
    expected_features: list[str],
) -> dict[str, str]:
    """Resolve source raster paths for each expected feature."""
    resolved = {
        _normalize_feature_name(feature): str(path)
        for feature, path in (explicit_sources or {}).items()
    }

    if source_dir is None:
        return resolved

    source_dir = Path(source_dir)
    for feature in expected_features:
        normalized = _normalize_feature_name(feature)
        if normalized in resolved:
            continue
        candidate = source_dir / f"{normalized}.tif"
        if candidate.exists():
            resolved[normalized] = str(candidate)
    return resolved


def _compare_arrays(
    stack_path: str | Path,
    stack_band: int,
    source_path: str | Path,
    *,
    sample_step: int | None = None,
) -> tuple[bool, bool, float | None]:
    """Return exact/same-grid/max-diff comparison for one band pair."""
    with rasterio.open(stack_path) as stack_src, rasterio.open(source_path) as src:
        same_grid = (
            stack_src.width == src.width
            and stack_src.height == src.height
            and stack_src.crs == src.crs
            and stack_src.transform.almost_equals(src.transform)
        )
        if not same_grid:
            return False, False, None

        stack_nodata = _band_nodata(stack_src, stack_band)
        source_nodata = _band_nodata(src, 1)
        max_abs_diff = None
        any_valid = False

        for window in _iter_windows(stack_src):
            stack_data = stack_src.read(stack_band, window=window, out_dtype=np.float32)
            source_data = src.read(1, window=window, out_dtype=np.float32)
            if sample_step is not None and sample_step > 1:
                stack_data = stack_data[::sample_step, ::sample_step]
                source_data = source_data[::sample_step, ::sample_step]
            stack_mask = _valid_mask(stack_data, stack_nodata)
            source_mask = _valid_mask(source_data, source_nodata)
            joint_mask = stack_mask & source_mask
            if not np.any(joint_mask):
                continue

            any_valid = True
            diffs = np.abs(stack_data[joint_mask] - source_data[joint_mask])
            block_max = float(diffs.max())
            max_abs_diff = block_max if max_abs_diff is None else max(max_abs_diff, block_max)

        if not any_valid:
            return False, True, None

        return max_abs_diff <= _FLOAT_TOLERANCE, True, max_abs_diff


def compare_stack_to_sources(
    stacked_inputs: str | Path,
    *,
    source_dir: str | Path | None = None,
    explicit_sources: dict[str, str] | None = None,
    expected_features: list[str] | None = None,
    sample_step: int | None = None,
) -> dict[str, Any]:
    """Compare stack bands against source rasters by order, stats, and array values."""
    stack_stats = _stack_band_stats(stacked_inputs)
    descriptions = [band.description for band in stack_stats]
    ordered_features = expected_features or descriptions
    ordered_features = [_normalize_feature_name(feature) for feature in ordered_features]

    source_paths = _resolve_expected_sources(
        source_dir,
        explicit_sources,
        ordered_features,
    )
    source_stats = {
        feature: _compute_band_stats(path)
        for feature, path in source_paths.items()
    }

    comparisons: list[BandComparison] = []
    for index, band_stats in enumerate(stack_stats):
        expected_feature = (
            ordered_features[index] if index < len(ordered_features) else band_stats.description
        )
        expected_source = source_paths.get(expected_feature)
        exact_match = None
        same_grid = False
        max_abs_diff = None

        if expected_source is not None:
            exact_match, same_grid, max_abs_diff = _compare_arrays(
                stacked_inputs,
                band_stats.band,
                expected_source,
                sample_step=sample_step,
            )

        best_feature = None
        best_score = None
        for feature_name, stats in source_stats.items():
            score = abs(band_stats.mean - stats.mean) + abs(band_stats.std - stats.std)
            if best_score is None or score < best_score:
                best_score = score
                best_feature = feature_name

        expected_stats = source_stats.get(expected_feature)
        comparisons.append(
            BandComparison(
                band=band_stats.band,
                description=band_stats.description,
                expected_feature=expected_feature,
                expected_source=expected_source,
                exact_match=exact_match,
                same_grid=same_grid,
                max_abs_diff=max_abs_diff,
                mean_diff=(
                    None if expected_stats is None else band_stats.mean - expected_stats.mean
                ),
                std_diff=(
                    None if expected_stats is None else band_stats.std - expected_stats.std
                ),
                best_matching_feature=best_feature,
                order_matches_description=band_stats.description == expected_feature,
            ),
        )

    missing_sources = [
        feature for feature in ordered_features
        if feature not in source_paths
    ]

    return {
        "stacked_inputs": str(stacked_inputs),
        "stack_band_descriptions": descriptions,
        "expected_features": ordered_features,
        "source_paths": source_paths,
        "missing_sources": missing_sources,
        "sample_step": sample_step,
        "stack_stats": [asdict(stats) for stats in stack_stats],
        "source_stats": {feature: asdict(stats) for feature, stats in source_stats.items()},
        "comparisons": [asdict(item) for item in comparisons],
    }


def _parse_feature_mapping(values: list[str]) -> dict[str, str]:
    """Parse repeated feature=path CLI arguments."""
    mappings: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Invalid --feature mapping {value!r}. Use --feature name=/path/to/raster.tif",
            )
        feature, path = value.split("=", 1)
        mappings[_normalize_feature_name(feature)] = path
    return mappings


def _format_float(value: float | None) -> str:
    """Format optional floats for concise text output."""
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def _print_report(report: dict[str, Any]) -> None:
    """Render a concise human-readable comparison report."""
    print(f"Stack: {report['stacked_inputs']}")
    print(f"Band descriptions: {report['stack_band_descriptions']}")
    print(f"Expected features: {report['expected_features']}")
    if report["missing_sources"]:
        print(f"Missing source rasters: {report['missing_sources']}")
    print("")
    print(
        "band | description | expected | exact | same_grid | max_abs_diff | mean_diff | std_diff | best_match",
    )
    print("-" * 110)
    for item in report["comparisons"]:
        print(
            f"{item['band']:>4} | "
            f"{item['description']:<14} | "
            f"{item['expected_feature']:<14} | "
            f"{str(item['exact_match']):<5} | "
            f"{str(item['same_grid']):<9} | "
            f"{_format_float(item['max_abs_diff']):>12} | "
            f"{_format_float(item['mean_diff']):>9} | "
            f"{_format_float(item['std_diff']):>8} | "
            f"{item['best_matching_feature']}"
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare stacked VRT bands against source feature rasters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stacked_inputs",
        type=str,
        required=True,
        help="Path to the stacked VRT or multiband raster.",
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default=None,
        help="Directory containing source rasters named <feature>.tif.",
    )
    parser.add_argument(
        "--feature",
        action="append",
        default=[],
        help="Explicit source mapping like --feature ndsm=/path/to/ndsm.tif.",
    )
    parser.add_argument(
        "--expected_features",
        nargs="+",
        default=None,
        help="Expected feature order if you want to override stack descriptions.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full report as JSON.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help=(
            "Compare raster values on a coarse grid sample instead of every pixel. "
            "Stats are still computed exactly."
        ),
    )
    parser.add_argument(
        "--sample_step",
        type=int,
        default=_DEFAULT_SAMPLE_STEP,
        help="Pixel stride used by --sample.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    report = compare_stack_to_sources(
        args.stacked_inputs,
        source_dir=args.source_dir,
        explicit_sources=_parse_feature_mapping(args.feature),
        expected_features=args.expected_features,
        sample_step=(args.sample_step if args.sample else None),
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    _print_report(report)


if __name__ == "__main__":
    main()
