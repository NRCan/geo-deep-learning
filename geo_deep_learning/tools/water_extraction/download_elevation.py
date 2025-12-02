"""Download elevation data from WCS services for AOI-based water extraction."""

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import fiona
import requests
from lightning.pytorch.cli import LightningArgumentParser

log = logging.getLogger(__name__)


@dataclass
class AOIDownloadConfig:
    """Configuration for downloading elevation data for an AOI."""

    aoi_path: str
    identifier: str = "dtm"
    out_folder: str | None = None
    resolution: int = 1

    def run(self) -> None:
        """Execute the download for the configured AOI."""
        out = self.out_folder or str(Path(self.aoi_path).parent)
        download_all_for_aoi(
            aoi_path=self.aoi_path,
            out_folder=out,
            identifiers=[self.identifier],
            resolution=self.resolution,
        )


def get_bbox(aoi_path: str) -> tuple[tuple[float, float, float, float], int]:
    """
    Extract bounding box and EPSG code from vector AOI file.

    Args:
        aoi_path: Path to vector AOI file (shapefile, etc.)

    Returns:
        Tuple containing (bounds tuple, epsg code)

    """
    with fiona.open(aoi_path, "r") as src:
        crs = src.crs
        bounds = src.bounds  # (minx, miny, maxx, maxy)
        epsg = int(crs["init"].split(":")[1]) if "init" in crs else int(crs["EPSG"])
    return bounds, epsg


def create_directory(path: str) -> None:
    """
    Create directory if it doesn't exist.

    Args:
        path: Directory path to create

    """
    Path(path).mkdir(parents=True, exist_ok=True)


def is_valid_geotiff(path: str) -> bool:
    """
    Check if a file is a valid GeoTIFF with at least one band.

    Args:
        path: Path to the GeoTIFF file to validate

    Returns:
        True if valid, False otherwise

    """
    import rasterio

    try:
        with rasterio.open(path) as src:
            return src.count > 0
    except (OSError, rasterio.errors.RasterioIOError) as e:
        log.warning("Invalid GeoTIFF at %s: %s", path, e)
        return False


def download_wcs_product(
    identifier: str,
    bbox: tuple[float, float, float, float],
    epsg: int,
    resolution: int,
    output_path: str,
) -> None:
    """
    Download a single elevation product from WCS service.

    Args:
        identifier: Product identifier (e.g., 'dtm', 'dsm', 'intensity')
        bbox: Bounding box as (minx, miny, maxx, maxy)
        epsg: EPSG code for coordinate reference system
        resolution: Pixel resolution in meters
        output_path: Where to save the downloaded GeoTIFF

    Raises:
        RuntimeError: If download fails with non-200 HTTP status

    """
    base_url = "https://datacube.services.geo.ca/ows/elevation"
    gridoffsets = f"{resolution},{-resolution}"

    boundingbox = ",".join(
        [
            str(bbox[0]),
            str(bbox[1]),
            str(bbox[2]),
            str(bbox[3]),
            f"urn:ogc:def:crs:EPSG::{epsg}",
        ],
    )

    gridorigin = f"{bbox[0]},{bbox[3]}"

    params = {
        "SERVICE": "WCS",
        "VERSION": "1.1.1",
        "REQUEST": "GetCoverage",
        "FORMAT": "image/geotiff",
        "IDENTIFIER": identifier,
        "BOUNDINGBOX": boundingbox,
        "GRIDBASECRS": f"urn:ogc:def:crs:EPSG::{epsg}",
        "GRIDCS": "urn:ogc:def:cs:OGC:0.0:Grid2dSquareCS",
        "GRIDTYPE": "urn:ogc:def:method:WCS:1.1:2dSimpleGrid",
        "GRIDORIGIN": gridorigin,
        "GRIDOFFSETS": gridoffsets,
    }

    http_ok = 200
    timeout_seconds = 300
    response = requests.get(
        base_url,
        params=params,
        verify=True,
        timeout=timeout_seconds,
    )
    if response.status_code != http_ok:
        error_msg = f"Failed to download {identifier}: HTTP {response.status_code}"
        raise RuntimeError(error_msg)

    with Path(output_path).open("wb") as f:
        f.write(response.content)
    log.info("Downloaded %s to %s", identifier, output_path)


def download_all_for_aoi(
    aoi_path: str,
    out_folder: str,
    identifiers: "Sequence[str]",
    resolution: int,
) -> None:
    """
    Download all elevation products for an AOI.

    Args:
        aoi_path: Path to vector AOI file
        out_folder: Directory where products will be saved
        identifiers: List of product identifiers to download
        resolution: Pixel resolution in meters

    """
    create_directory(out_folder)
    bbox, epsg = get_bbox(aoi_path)

    for identifier in identifiers:
        out_path = Path(out_folder) / f"{identifier}.tif"
        if out_path.exists() and is_valid_geotiff(str(out_path)):
            log.info("[SKIP] %s already exists and is valid.", identifier)
            continue

        try:
            download_wcs_product(identifier, bbox, epsg, resolution, str(out_path))
        except (RuntimeError, requests.RequestException):
            log.exception("Failed to download %s for %s", identifier, aoi_path)
        time.sleep(2)


if __name__ == "__main__":
    parser = LightningArgumentParser()
    parser.add_class_arguments(AOIDownloadConfig, nested_key=None)
    args = parser.parse_args()
    config = AOIDownloadConfig(**vars(args))
    config.run()
