# Licensed under the MIT License.
# Authors: Victor Alhassan, Rémi Tavon

# Hardware requirements: 64 Gb RAM (cpu), 8 Gb GPU RAM.

"""CCMEO postprocess script."""
import codecs
import os
import shutil
from typing import Dict, Union, Sequence
from collections import OrderedDict
from pathlib import Path

import docker
import fiona
import geopandas
import rasterio
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from rasterio import features
from rasterio.windows import Window
from spython.main import Client
from tqdm import tqdm

from inference.InferenceDataset import InferenceDataset
from utils.logger import get_logger
from utils.utils import get_key_def

# Set the logging file
logging = get_logger(__name__)


def regularize_buildings(in_pred,
                         out_pred,
                         image,
                         container_type,
                         command,
                         building_value=1,
                         fallback: bool = True):
    """
    TODO
    @param in_pred:
    @param out_pred:
    @param image:
    @param container_type:
    @param command:
    @param building_value:
    @param fallback:
    @return:
    """
    try:  # will raise Exception if image is None --> default to ras2vec
        run_from_container(image='remtav/gdl', command=command,  # FIXME softcode reg image
                           binds={f"{str(in_pred.parent.absolute())}": "/home",
                                  f"/home/remi/PycharmProjects/projectRegularization/regularization": "/media"},
                           container_type=container_type)
        logging.info(f'\nRegularization completed')
    except Exception as e:
        logging.error(f"\nError regularizing using {container_type} container with image {image}."
                      f"\ncommand: {command}"
                      f"\nError {type(e)}: {e}"
                      f"\nWill try regularizing without container...")
        if fallback:
            try:
                from regularization import regularize
                regularize.main(
                    in_raster=in_pred,
                    out_raster=out_pred,
                    build_val=building_value,
                    models_dir="/home/remi/PycharmProjects/projectRegularization/regularization/saved_models_gan"
                    # TODO softcode
                )
            except ImportError:
                logging.critical(f"Failed to regularize buildings")


def polygonize(in_raster, 
               out_vector, 
               container_image: str = None, 
               container_type: str = 'docker', 
               container_command: str = '',
               fallback: bool = True):
    logging.info(f"Polygonizing prediction to {out_vector}...")
    try:  # will raise Exception if image, command or container type is None --> fallback to ras2vec
        run_from_container(image=container_image, command=container_command,
                           binds={f"{str(in_raster.parent.absolute())}": "/home"},
                           container_type=container_type)
    except Exception as e:
        logging.error(f"\nError polygonizing using {container_type} container with image {container_image}."
                      f"\nCommand: {container_command}"
                      f"\nError {type(e)}: {e}")
        if fallback:
            ras2vec(in_raster, out_vector)

    if out_vector.is_file():
        logging.info(f'\nPolygonization completed. Raw prediction: {out_vector}')
    else:
        logging.critical(f'\nPolygonization failed. Raw prediction not found: {out_vector}')
        raise FileNotFoundError(f'Raw prediction not found: {out_vector}')

    # in some cases, Grass fails to read source epsg and writes output without crs
    with rasterio.open(in_raster) as src:
        out_vect_no_crs = out_vector.parent / f"{out_vector.stem}_no_crs.gpkg"
        try:
            gdf = geopandas.read_file(out_vector)
        except fiona.errors.DriverError as e:
            logging.critical(f"\nOutputted polygonized prediction may be empty."
                             f"\n{type(e)}: {e}"
                             f"\nSkipping all remaining postprocessing and exiting...")
            return
        if gdf.crs != src.crs:
            shutil.copy(out_vector, out_vect_no_crs)
            gdf = geopandas.read_file(out_vect_no_crs)
            logging.warning(f"Outputted polygonized prediction may not have valid CRS. Will attempt to set it to"
                            f"raster prediction's CRS")
            gdf.set_crs(crs=src.crs, allow_override=True)
            # write file
            gdf.to_file(out_vector)
            os.remove(out_vect_no_crs)


def run_from_container(image: str, command: str, binds: Dict = {}, container_type='docker', verbose: bool = True):
    """
    Runs a command inside a docker or singularity container and returns when container has exited (on success or fail)
    @param image: str
        Name or path to image to use
    @param command: str
        Command to pass to run/exec
    @param binds: dict
        A dictionary to configure volumes mounted inside the container.
        The key is either the host path or a volume name, and the value is a container path.
        Currently hardcoded to mount the volumes read/write
    @param container_type: str
        Specifies whether to use "docker" or "singularity"
    @param verbose: bool
        if True, will print logs as container if executed
    @return:
    """
    stream = True if verbose else False
    logging.debug(command)
    if container_type == 'docker':
        binds = {k: {'bind': v, 'mode': 'rw'} for k, v in binds.items()}
        client = docker.from_env()
        qgis_pp_docker_img = client.images.get(image)
        container = client.containers.run(
            image=qgis_pp_docker_img,
            command=command,
            volumes=binds,
            device_requests=[
                docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
            ],
            detach=True)
        logs = container.logs(stream=stream)
        if stream:
            for line in logs:
                logging.info(codecs.decode(line))
        exit_code = container.wait(timeout=1200)
        logging.info(exit_code)
    # Singularity: validate installation and assert version >= 3.0.0
    elif container_type == 'singularity' and \
            Client.version() and int(Client.version().split(' ')[-1].split('.')[0]) >= 3:
        binds = [f"{k}:{v}" for k, v in binds.items()]
        logging.debug(command.split(" "))
        logs = Client.execute(to_absolute_path(str(image)), command.split(" "),
                              bind=binds, stream=stream)
        if stream:
            for line in logs:
                logging.info(codecs.decode(line))
    else:
        logging.error(f"\nContainer type is not valid. Choose 'docker' or 'singularity'")


def ras2vec(raster_file, output_path):
    """
    Polygonizes a raster prediction from a semantic segmentation models, after argmax() operation.
    @param raster_file: str, file object or pathlib.Path object
        Path to input raster prediction to be polygonized
    @param output_path: str, file object or pathlib.Path object
        Path to output vector prediction to be polygonized
    @return:
    """
    logging.warning(f"\nUsing rasterio and shapely. This method is less efficient than Grass' r.to.vect."
                    f"\nIf possible, use container approach with Grass.")
    # Create a generic polygon schema for the output vector file
    i = 0
    feat_schema = {'geometry': 'Polygon',
                   'properties': OrderedDict([('value', 'int')])
                   }
    class_value_domain = set()
    out_features = []

    logging.info("\n   - Processing raster file: {}".format(raster_file))
    with rasterio.open(raster_file, 'r') as src:
        raster = src.read(1)
    mask = raster != 0
    # Vectorize the polygons
    polygons = features.shapes(raster, mask, transform=src.transform)

    # Create shapely polygon features
    for polygon in polygons:
        feature = {'geometry': {
            'type': 'Polygon',
            'coordinates': None},
            'properties': OrderedDict([('value', 0)])}

        feature['geometry']['coordinates'] = polygon[0]['coordinates']
        value = int(polygon[1])  # Pixel value of the class (layer)
        class_value_domain.add(value)
        feature['properties']['value'] = value
        i += 1
        out_features.append(feature)

    logging.info("\n   - Writing output vector file: {}".format(output_path))
    num_layers = list(class_value_domain)  # Number of unique pixel value
    if not num_layers:
        logging.critical(f"\nRaster prediction contains only background pixels. An empty output will be written")
        fiona.open(output_path, 'w', crs=src.crs, layer='empty', schema=feat_schema, driver='GPKG')
        return
    for num_layer in num_layers:
        polygons = [feature for feature in out_features if feature['properties']['value'] == num_layer]
        layer_name = 'vector_' + str(num_layer).rjust(3, '0')
        logging.info("   - Writing layer: {}".format(layer_name))

        with fiona.open(output_path, 'w',
                        crs=src.crs,
                        layer=layer_name,
                        schema=feat_schema,
                        driver='GPKG') as dest:
            for polygon in polygons:
                dest.write(polygon)
    logging.info("\nNumber of features written: {}".format(i))


def add_confidence_from_heatmap(in_heatmap: Union[str, Path], in_vect: Union[str, Path]):
    """
    Writes confidence values to polygonized prediction from heatmap outputted by pytorch semantic segmentation models.
    @param in_heatmap: str, file object or pathlib.Path object
        Path to raster as outputted by pytorch semantic segmentation models. Expects a 3D raster with
        (number of classes, width, height). Confidence values should be integers. These values will be taken directly
        from heatmap. Therefore, values should be between 0 and 100 (ex.: softmax * 100) if a percentage value is
        desired. Naturally, the heatmap must be outputted before any argmax() operation in order to keep per-pixel,
        per-class confidence values from raw prediction.
    @param in_vect: str, file object or pathlib.Path object
        Path to input polygonized version of raw raster prediction. Class values should be written to "value" column
        with classes as continuous integers starting at 0 (0 being background class for multi-class tasks).
        This function will iterate over predicted features. For each feature, a corresponding window is read in
        heatmap, mean confidence is computted, then written to output GeoDataFrame
        See polygonization tools used in this script (ex.: Grass' r.to.vect)
    @return:
    """
    in_vect_copy = in_vect.parent / f"{in_vect.stem}_no_confid.gpkg"
    shutil.copy(in_vect, in_vect_copy)
    with rasterio.open(in_heatmap, 'r') as src:
        gdf = geopandas.read_file(in_vect_copy)
        if gdf.empty:
            logging.warning(f"\nNo features in polygonized output. No confidence values can be written.")
            return
        confidences = []
        for row, bbox in tqdm(zip(gdf.iterrows(), gdf.envelope),
                              desc=f"Calculating confidence values per feature",
                              total=len(gdf)):
            index, feature = row
            left, bottom, right, top = bbox.bounds
            window = rasterio.windows.from_bounds(left, bottom, right, top, transform=src.transform)
            conf_vals = src.read(window=window)  # only read band relative to feature class
            mask = conf_vals.argmax(axis=0) == feature.value  # TODO test with single-class heatmaps
            confidence = conf_vals[feature.value][mask].mean()
            confidence = round(confidence.astype(int)) if confidence >= 0 else None
            confidences.append(confidence)
        gdf['confidence'] = confidences
        gdf.to_file(in_vect)
        os.remove(in_vect_copy)
        logging.info(f'\nConfidence values added to polygonized prediction. Output: {in_vect}')


def main(params):
    """High-level postprocess pipeline.
    Runs building regularization, polygonization and generalization of a raster prediction and saves results to gpkg.
    Args:
        params: configuration parameters
    """
    in_name = get_key_def('input_name', params['postprocess'], expected_type=str)
    root = get_key_def('root_dir', params['postprocess'], default="data", to_path=True, validate_path_exists=True)

    # Post-processing
    confidence_values = get_key_def('confidence_values', params['postprocess'], expected_type=bool, default=True)
    regularization = get_key_def('regularization', params['postprocess'], expected_type=bool, default=True)
    generalization = get_key_def('generalization', params['postprocess'], expected_type=bool, default=True)

    # output suffixes
    out_reg_suffix = get_key_def('regularization', params['postprocess']['output_suffixes'], default='_reg',
                                  expected_type=str)
    out_poly_suffix = get_key_def('polygonization', params['postprocess']['output_suffixes'], default='_raw',
                                  expected_type=str)
    out_gen_suffix = get_key_def('generalization', params['postprocess']['output_suffixes'], default='_post',
                                  expected_type=str)

    # regularization container parameters
    reg_fallback = get_key_def('fallback', params['postprocess']['reg_cont'], expected_type=bool, default=True)
    reg_cont_type = get_key_def('cont_type', params['postprocess']['reg_cont'], expected_type=str)
    reg_cont_image = get_key_def('cont_image', params['postprocess']['reg_cont'], expected_type=str)
    reg_command = get_key_def('command', params['postprocess']['reg_cont'], expected_type=str)
    
    # polygonization container parameters
    poly_fallback = get_key_def('fallback', params['postprocess']['poly_cont'], expected_type=bool, default=True)
    poly_cont_type = get_key_def('cont_type', params['postprocess']['poly_cont'], expected_type=str)
    poly_cont_image = get_key_def('cont_image', params['postprocess']['poly_cont'], expected_type=str)
    poly_command = get_key_def('command', params['postprocess']['poly_cont'], expected_type=str)

    # generalization container parameters
    qgis_models_dir = get_key_def('qgis_models_dir', params['postprocess']['gen_cont'], expected_type=str, validate_path_exists=True)
    gen_cont_type = get_key_def('cont_type', params['postprocess']['gen_cont'], expected_type=str)
    gen_cont_image = get_key_def('cont_image', params['postprocess']['gen_cont'], expected_type=str)
    gen_commands = get_key_def('command', params['postprocess']['gen_cont'], expected_type=DictConfig)

    # filter generalization commands based on extracted classes
    classes_dict = get_key_def('classes_dict', params['dataset'], expected_type=DictConfig)
    gen_cmds_pruned = {data_class: cmd for data_class, cmd in gen_commands.items() if data_class in classes_dict.keys()}

    # build inputs paths and check building value expected from model
    outpath = root / f"{in_name}.tif"
    if not outpath.is_file():
        raise FileNotFoundError(f"\nCannot find raster prediction file to use for postprocessing."
                                f"\nGot:{outpath}")
    in_heatmap = root / f"{in_name}_heatmap.tif"
    building_value = classes_dict['BUIL']

    # build output paths
    out_reg = root / f"{in_name}{out_reg_suffix}.tif"
    out_poly = root / f"{in_name}{out_poly_suffix}.gpkg"
    out_gen = root / f"{in_name}{out_gen_suffix}.gpkg"

    if regularization:  # TODO: adapt regularization to process from vector to vector?
        logging.info(f"Regularizing prediction. Polygonized output will be overwritten."
                     f"\nOutput: {out_poly}")
        regularize_buildings(
            in_pred=outpath,
            out_pred=out_reg,
            image=reg_cont_image,
            container_type=reg_cont_type,
            command=reg_command,
            building_value=building_value,
            fallback=reg_fallback,
        )
        # TODO: assumes knowledge of command from config
        if out_reg.is_file():
            poly_command = poly_command.replace(f"{in_name}.tif", f"{out_reg.stem}.tif")
            outpath = outpath.parent / f"{out_reg.stem}.tif"

    # Postprocess final raster prediction (polygonization)
    polygonize(
        in_raster=outpath,
        out_vector=out_poly,
        container_image=poly_cont_image,
        container_type=poly_cont_type,
        container_command=poly_command,
        fallback=poly_fallback,
    )

    # set confidence values to features in polygonized prediction
    if confidence_values and in_heatmap.is_file():
        add_confidence_from_heatmap(in_heatmap=in_heatmap, in_vect=out_poly)
    elif confidence_values:
        logging.error(f"Cannot add confidence levels to polygons. A heatmap must be generated at inference")

    if generalization:
        logging.info(f"Generalizing prediction to {out_gen}")
        for command in gen_cmds_pruned.values():
            try:  # will raise Exception if image is None --> default to ras2vec
                run_from_container(image=gen_cont_image, command=command,
                                   binds={f"{str(outpath.parent.absolute())}": "/home",
                                          f"{str(qgis_models_dir)}": "/models"},
                                   container_type=gen_cont_type)
            except Exception as e:
                logging.error(f"\nError generalizing using {gen_cont_type} container with image {gen_cont_image}."
                              f"\ncommand: {command}"
                              f"\nError {type(e)}: {e}")
        if out_gen.is_file():
            logging.info(f'\nGeneralization completed. Final prediction: {out_gen}')
        else:
            logging.error(f'\nGeneralization failed. See logs...')
            raise FileNotFoundError(f"{out_gen}")

    logging.info(f'\nEnd of postprocessing')
