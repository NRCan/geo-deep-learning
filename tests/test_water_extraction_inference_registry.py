"""Tests for model-driven inference registry support."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin

from geo_deep_learning.tools.water_extraction import inference


def _write_single_band_raster(
    path: Path,
    data: np.ndarray,
    *,
    dtype: str = "float32",
    nodata: float | int | None = -32767,
    description: str | None = None,
) -> None:
    """Write a small single-band raster for testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=dtype,
        transform=from_origin(0, data.shape[0], 1, 1),
        crs="EPSG:4326",
        nodata=nodata,
    ) as dst:
        dst.write(data.astype(dtype), 1)
        if description is not None:
            dst.set_band_description(1, description)


def test_resolve_model_config_rejects_unknown_features(tmp_path: Path) -> None:
    """Unknown feature names should fail validation."""
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"placeholder")
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "bad_model": {
                        "checkpoint": str(checkpoint),
                        "features": ["ndsm", "bogus_feature"],
                        "mean": [0.0, 0.0],
                        "std": [1.0, 1.0],
                    },
                },
            },
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown feature"):
        inference._resolve_model_config("bad_model", str(registry_path))


def test_resolve_model_config_requires_existing_checkpoint(tmp_path: Path) -> None:
    """Registry validation should fail when the checkpoint path is missing."""
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "missing_ckpt": {
                        "checkpoint": str(tmp_path / "missing.ckpt"),
                        "features": ["ndsm", "twi"],
                        "mean": [0.0, 0.0],
                        "std": [1.0, 1.0],
                    },
                },
            },
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="does not exist"):
        inference._resolve_model_config("missing_ckpt", str(registry_path))


def test_resolve_model_config_rejects_stats_length_mismatch(tmp_path: Path) -> None:
    """Registry stats must match the configured feature count."""
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"placeholder")
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "bad_stats": {
                        "checkpoint": str(checkpoint),
                        "features": ["ndsm", "twi", "intensity"],
                        "mean": [0.0, 0.0],
                        "std": [1.0, 1.0, 1.0],
                    },
                },
            },
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="stats length mismatch"):
        inference._resolve_model_config("bad_stats", str(registry_path))


def test_preprocess_aoi_builds_requested_features_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requested features should be built once and stacked in registry order."""
    raw_aoi = tmp_path / "inference" / "raw" / "AOI_001"
    preprocessed_root = tmp_path / "inference" / "preprocessed"
    dtm = np.array([[1, 2], [3, 4]], dtype=np.float32)
    dsm = np.array([[5, 6], [7, 8]], dtype=np.float32)
    intensity = np.array([[0, 1], [2, 0]], dtype=np.float32)

    _write_single_band_raster(raw_aoi / "dtm.tif", dtm, description="dtm")
    _write_single_band_raster(raw_aoi / "dsm.tif", dsm, description="dsm")
    _write_single_band_raster(
        raw_aoi / "intensity.tif",
        intensity,
        description="intensity",
    )

    def fake_align_to_reference(
        reference_path: str,
        input_path: str,
        output_path: str,
        nodata_val: float,
    ) -> None:
        del reference_path, nodata_val
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(input_path, output_path)

    def fake_compute_ndsm(
        dsm_path: str,
        dtm_path: str,
        output_path: str,
        nodata_val: float,
    ) -> None:
        del dsm_path, dtm_path, nodata_val
        _write_single_band_raster(
            Path(output_path),
            np.full((2, 2), 10, dtype=np.float32),
            description="ndsm",
        )

    def fake_compute_twi(
        dtm_path: str,
        twi_output_path: str,
        keep_intermediate: bool,
    ) -> None:
        del dtm_path, keep_intermediate
        _write_single_band_raster(
            Path(twi_output_path),
            np.full((2, 2), 20, dtype=np.float32),
            description="twi",
        )

    def fake_build_zero_intensity_mask(
        intensity_path: Path,
        *,
        min_component_size: int = 1,
        max_hole_size: int = 0,
    ) -> tuple[np.ndarray, dict]:
        del min_component_size, max_hole_size
        with rasterio.open(intensity_path) as src:
            profile = src.profile.copy()
        profile.update(dtype="uint8", nodata=255, count=1, compress="lzw")
        return np.array([[1, 0], [0, 1]], dtype=np.uint8), profile

    monkeypatch.setattr(inference, "align_to_reference", fake_align_to_reference)
    monkeypatch.setattr(inference, "compute_ndsm", fake_compute_ndsm)
    monkeypatch.setattr(inference, "compute_twi_whitebox", fake_compute_twi)
    monkeypatch.setattr(
        inference,
        "build_zero_intensity_mask",
        fake_build_zero_intensity_mask,
    )

    stacked_path = inference.preprocess_aoi(
        data_folder=str(raw_aoi),
        output_folder=str(preprocessed_root),
        requested_features=["ndsm", "intensity", "zero_intensity"],
    )

    assert Path(stacked_path).exists()
    with rasterio.open(stacked_path) as src:
        assert src.count == 3
        assert src.descriptions == ("ndsm", "intensity", "zero_intensity")

    assert (preprocessed_root / "AOI_001" / "zero_intensity.tif").exists()

    second_path = inference.preprocess_aoi(
        data_folder=str(raw_aoi),
        output_folder=str(preprocessed_root),
        requested_features=["ndsm", "intensity", "zero_intensity"],
    )
    assert second_path == stacked_path


def test_preprocess_aoi_rejects_missing_intensity_dependency(tmp_path: Path) -> None:
    """Intensity-derived features should fail fast when intensity.tif is absent."""
    raw_aoi = tmp_path / "inference" / "raw" / "AOI_002"
    preprocessed_root = tmp_path / "inference" / "preprocessed"
    dtm = np.array([[1, 2], [3, 4]], dtype=np.float32)
    dsm = np.array([[5, 6], [7, 8]], dtype=np.float32)

    _write_single_band_raster(raw_aoi / "dtm.tif", dtm, description="dtm")
    _write_single_band_raster(raw_aoi / "dsm.tif", dsm, description="dsm")

    with pytest.raises(FileNotFoundError, match="Requested feature\\(s\\)"):
        inference.preprocess_aoi(
            data_folder=str(raw_aoi),
            output_folder=str(preprocessed_root),
            requested_features=["ndsm", "zero_intensity"],
        )


def test_main_uses_registry_model_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The automated CLI path should forward registry-derived settings."""
    captured: dict[str, object] = {}

    def fake_run_inference(**kwargs: object) -> dict[str, str]:
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(inference, "run_inference", fake_run_inference)
    monkeypatch.setattr(
        inference,
        "_resolve_model_config",
        lambda model_id, registry_path: {
            "model_id": model_id,
            "checkpoint": str(tmp_path / "model.ckpt"),
            "features": ["ndsm", "intensity", "zero_intensity"],
            "mean": [1.0, 2.0, 3.0],
            "std": [4.0, 5.0, 6.0],
            "threshold": 0.7,
            "channel_weights": [1.0, 1.1, 1.2],
            "output_name": "registry_model",
        },
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inference.py",
            "--model_id",
            "model_4",
            "--registry",
            str(tmp_path / "registry.yaml"),
            "--stacked_inputs",
            str(tmp_path / "stacked_inputs.tif"),
            "--output_folder",
            str(tmp_path / "results"),
        ],
    )

    inference.main()

    assert captured["checkpoint_path"] == str(tmp_path / "model.ckpt")
    assert captured["requested_features"] == ["ndsm", "intensity", "zero_intensity"]
    assert captured["mean"] == [1.0, 2.0, 3.0]
    assert captured["std"] == [4.0, 5.0, 6.0]
    assert captured["binary_threshold"] == 0.7
    assert captured["channel_weights"] == [1.0, 1.1, 1.2]
    assert captured["include_intensity"] is True
    assert captured["selected_channel_indices"] is None
    assert captured["output_name"] == "registry_model"
