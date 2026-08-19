"""Tests for elevation download helpers."""

from __future__ import annotations

import pytest

from geo_deep_learning.tools.water_extraction import download_elevation


def test_estimate_output_shape_rounds_up() -> None:
    """Output shape should cover the full AOI envelope."""
    height, width = download_elevation._estimate_output_shape(
        (0.0, 0.0, 10.2, 20.7),
        resolution=2,
    )
    assert (height, width) == (11, 6)


def test_validate_output_size_rejects_large_aoi(monkeypatch: pytest.MonkeyPatch) -> None:
    """Large AOIs should fail fast before any expensive downloads begin."""
    monkeypatch.setenv("GDL_MAX_ELEVATION_OUTPUT_PIXELS", "100")

    with pytest.raises(RuntimeError, match="too large for the current in-memory merge path"):
        download_elevation._validate_output_size(
            identifier="dtm",
            aoi_bounds=(0.0, 0.0, 100.0, 100.0),
            resolution=1,
        )


def test_validate_output_size_accepts_small_aoi(monkeypatch: pytest.MonkeyPatch) -> None:
    """Small AOIs should pass the output-size guard."""
    monkeypatch.setenv("GDL_MAX_ELEVATION_OUTPUT_PIXELS", "10000")

    height, width, pixel_count = download_elevation._validate_output_size(
        identifier="dtm",
        aoi_bounds=(0.0, 0.0, 50.0, 40.0),
        resolution=2,
    )

    assert (height, width, pixel_count) == (20, 25, 500)
