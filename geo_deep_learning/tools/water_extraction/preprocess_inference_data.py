"""
CLI for preprocessing AOI for inference (no model required).

Usage:
    python -m geo_deep_learning.tools.water_extraction.preprocess_inference_data \
        --data_folder data/inference/raw/<aoi> \
        --output_folder data/inference/preprocessed
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
        help=(
            "Path to inference AOI raw folder with dtm.tif, dsm.tif, intensity.tif "
            "(e.g., data/inference/raw/02NB000)"
        ),
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help=(
            "Path to inference preprocessed root "
            "(e.g., data/inference/preprocessed)"
        ),
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
        "--seam_sigma",
        type=float,
        default=1.5,
        help=(
            "Gaussian sigma for seam correction inpainting in pixels (default 1.5). "
            "Should be at least seam_width_pixels to bridge the inpainting gap."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the AOI preprocessing pipeline from command line arguments."""
    args = parse_args()
    preprocess_aoi(
        data_folder=args.data_folder,
        output_folder=args.output_folder,
        workflow="inference",
        include_intensity=not args.no_intensity,
        project_extents_path=args.project_extents,
        seam_gaussian_sigma=args.seam_sigma,
    )
    log.info(
        "Preprocessing complete. Outputs in: %s/%s",
        args.output_folder,
        args.data_folder.rstrip("/").split("/")[-1],
    )


if __name__ == "__main__":
    main()
