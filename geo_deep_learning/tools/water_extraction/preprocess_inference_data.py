"""
CLI for preprocessing AOI for inference (no model required).

Usage:
    python -m geo_deep_learning.tools.water_extraction.preprocess_inference_data \
        --data_folder path/to/aoi \
        --output_folder path/to/outputs
"""

import argparse
import logging

from geo_deep_learning.tools.water_extraction.inference import preprocess_aoi

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the preprocessing CLI."""
    parser = argparse.ArgumentParser(
        description="Preprocess AOI for inference (alignment, nDSM, TWI, stacking)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        required=True,
        help="Path to AOI folder with dtm.tif, dsm.tif, intensity.tif",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Where to save preprocessed outputs",
    )
    parser.add_argument(
        "--no_intensity",
        action="store_true",
        help="Exclude intensity band from stack",
    )
    parser.add_argument(
        "--project_extents",
        type=str,
        default=None,
        help=(
            "Path to GeoPackage with one polygon per LiDAR project extent. "
            "When provided, seam correction is applied to the DTM and DSM before "
            "any derivatives are computed. Omit to skip seam correction."
        ),
    )
    parser.add_argument(
        "--seam_sigma_color",
        type=float,
        default=0.3,
        help=(
            "Bilateral range-kernel sigma for seam correction, in data units (metres "
            "for DTM/DSM). Default 0.3 m is appropriate for elevation rasters."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the AOI preprocessing pipeline from command line arguments."""
    args = parse_args()
    preprocess_aoi(
        data_folder=args.data_folder,
        output_folder=args.output_folder,
        include_intensity=not args.no_intensity,
        project_extents_path=args.project_extents,
        seam_sigma_color=args.seam_sigma_color,
    )
    log.info("Preprocessing complete. Outputs in: %s", args.output_folder)


if __name__ == "__main__":
    main()
