# Licensed under the MIT License.
# Authors: Victor Alhassan, Rémi Tavon

# Hardware requirements: 64 Gb RAM (cpu), 8 Gb GPU RAM.

"""CCMEO validation script."""
import os
import re
import shutil

import fiona
import geopandas
import numpy as np
import rasterio
from omegaconf import DictConfig

from models.model_choice import read_checkpoint
from utils.logger import get_logger
from utils.utils import get_key_def, override_model_params_from_checkpoint, extension_remover, ckpt_is_compatible

# Set the logging file
logging = get_logger(__name__)


def main(params):
    """High-level postprocess pipeline.
    Runs building regularization, polygonization and generalization of a raster prediction and output a GeoPackage
    @param params:
        Pipeline configuration parameters
    """
    root = get_key_def('root_dir', params['postprocess'], default="data", to_path=True, validate_path_exists=True)
    models_dir = get_key_def('checkpoint_dir', params['inference'], default='checkpoints', to_path=True,
                             validate_path_exists=True)
    inf_outname = get_key_def('output_name', params['inference'], expected_type=str)
    if not inf_outname:
        raise ValueError(f"\nNo inference output name is set. This parameter is required during postprocessing for "
                         f"\nhydra's successful interpolation of input and output names in commands set in config.")
    inf_outname = extension_remover(inf_outname)
    inf_outpath = root / f"{inf_outname}.tif"
    outname = get_key_def('output_name', params['postprocess'], default=inf_outname, expected_type=str)
    outname = extension_remover(outname)
    if not inf_outpath.is_file():
        raise FileNotFoundError(f"\nCannot find raster prediction file to use for postprocessing."
                                f"\nGot:{inf_outpath}")
    checkpoint = get_key_def('state_dict_path', params['postprocess'], expected_type=str, to_path=True,
                             validate_path_exists=True)

    # Post-processing
    gen_commands = dict(get_key_def('command', params['postprocess']['gen_cont'], expected_type=DictConfig))
    # output suffixes
    out_gen_suffix = get_key_def('generalization', params['postprocess']['output_suffixes'], default='_post',
                                  expected_type=str)
    if not ckpt_is_compatible(checkpoint):
        raise KeyError(f"\nCheckpoint is incompatible with inference pipeline.")
    checkpoint_dict = read_checkpoint(checkpoint, out_dir=models_dir)
    params = override_model_params_from_checkpoint(params=params, checkpoint_params=checkpoint_dict['params'])

    # filter generalization commands based on extracted classes
    classes_dict = get_key_def('classes_dict', params['dataset'], expected_type=DictConfig)
    classes_dict = {k: v for k, v in classes_dict.items() if v}  # Discard keys where value is None
    gen_cmds_pruned = {data_class: cmd for data_class, cmd in gen_commands.items() if data_class in classes_dict.keys()}
    out_layers = []
    for cmd in gen_cmds_pruned.values():
        results = re.findall(r'--outlayername[\w]*=[\w]*', cmd)
        [out_layers.append(result.split('=')[-1]) for result in results]
    
    # build output paths
    out_gen = root / f"{outname}{out_gen_suffix}.gpkg"

    # in some cases, Grass fails to read source epsg and writes output without crs
    with rasterio.open(inf_outpath) as src:
        out_vect_no_crs = out_gen.parent / f"{out_gen.stem}_no_crs.gpkg"
        try:
            gdf = geopandas.read_file(out_gen)
        except fiona.errors.DriverError as e:
            logging.critical(f"\nOutputted polygonized prediction may be empty.")
            raise e
        if gdf.crs != src.crs:
            shutil.copy(out_gen, out_vect_no_crs)
            gdf = geopandas.read_file(out_vect_no_crs)
            logging.warning(f"Outputted polygonized prediction may not have valid CRS. Will attempt to set it to"
                            f"raster prediction's CRS")
            gdf.set_crs(crs=src.crs, allow_override=True)
            # write file
            gdf.to_file(out_gen)
            os.remove(out_vect_no_crs)

    # Does output exist?
    if not out_gen.is_file():
        raise FileNotFoundError(f'Raw prediction not found: {out_gen}')

    for layer in out_layers:
        logging.info(f'Validating layer "{layer}"...')
        gdf = geopandas.read_file(out_gen, layer=layer)
        if gdf.empty:
            logging.critical(f"\nOutput prediction {out_gen} contains no features in layer {layer}.")
        if not np.all(gdf.is_valid.values):
            raise AssertionError(f"\nOutput prediction {out_gen} contains an invalid geometry in layer {layer}.")
        else:
            logging.info(f"Found {len(gdf)} valid features")

    logging.info(f'\nEnd of validation')
