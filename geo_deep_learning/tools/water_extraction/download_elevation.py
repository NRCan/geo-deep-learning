import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import fiona
import requests
from lightning.pytorch.cli import LightningArgumentParser


@dataclass
class AOIDownloadConfig:
    aoi_path: str
    identifier: str = "dtm"
    out_folder: str | None = None
    resolution: int = 1

    def run(self) -> None:
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
    import rasterio

    try:
        with rasterio.open(path) as src:
            return src.count > 0
    except Exception as e:
        print(f"Invalid GeoTIFF at {path}: {e}")
        return False


def download_wcs_product(
    identifier: str,
    bbox: tuple[float, float, float, float],
    epsg: int,
    resolution: int,
    output_path: str,
) -> None:
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

    response = requests.get(base_url, params=params, verify=False)
    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to download {identifier}: HTTP {response.status_code}",
        )

    with Path(output_path).open("wb") as f:
        f.write(response.content)
    print(f"Downloaded {identifier} to {output_path}")


def download_all_for_aoi(
    aoi_path: str,
    out_folder: str,
    identifiers: "Sequence[str]",
    resolution: int,
) -> None:
    create_directory(out_folder)
    bbox, epsg = get_bbox(aoi_path)

    for identifier in identifiers:
        out_path = Path(out_folder) / f"{identifier}.tif"
        if out_path.exists() and is_valid_geotiff(str(out_path)):
            print(f"[SKIP] {identifier} already exists and is valid.")
            continue

        try:
            download_wcs_product(identifier, bbox, epsg, resolution, str(out_path))
        except Exception as e:
            print(f"[ERROR] Failed to download {identifier} for {aoi_path}: {e}")
        time.sleep(2)


if __name__ == "__main__":
    parser = LightningArgumentParser()
    parser.add_class_arguments(AOIDownloadConfig, nested_key=None)
    args = parser.parse_args()
    config = AOIDownloadConfig(**vars(args))
    config.run()
