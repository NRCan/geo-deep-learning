"""CLI for preprocessing AOI for inference (no model required).

Usage:
    python -m geo_deep_learning.tools.water_extraction.preprocess_inference_data \
        --data_folder path/to/aoi \
        --output_folder path/to/outputs
"""
import argparse
import logging
from pathlib import Path
from geo_deep_learning.tools.water_extraction.inference import preprocess_aoi

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess AOI for inference (alignment, nDSM, TWI, stacking)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_folder", type=str, required=True, help="Path to AOI folder with dtm.tif, dsm.tif, intensity.tif")
    parser.add_argument("--output_folder", type=str, required=True, help="Where to save preprocessed outputs")
    parser.add_argument("--no_intensity", action="store_true", help="Exclude intensity band from stack")
    return parser.parse_args()

def main():
    args = parse_args()
    preprocess_aoi(
        data_folder=args.data_folder,
        output_folder=args.output_folder,
        include_intensity=not args.no_intensity,
    )
    log.info("Preprocessing complete. Outputs in: %s", args.output_folder)

if __name__ == "__main__":
    main()
