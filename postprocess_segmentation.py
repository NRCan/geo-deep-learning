# Licensed under the MIT License.
# Authors: Victor Alhassan, Rémi Tavon

# Hardware requirements: 64 Gb RAM (cpu), 8 Gb GPU RAM.

"""CCMEO postprocess script."""
import codecs
from typing import Dict
from collections import OrderedDict
from pathlib import Path

import docker
import fiona
import geopandas
import numpy as np
import rasterio
from fiona.crs import to_string
from hydra.utils import to_absolute_path
from rasterio import features
from rasterio.windows import Window
from spython.main import Client

from inference.InferenceDataModule import InferenceDataModule
from utils.logger import get_logger
from utils.utils import get_key_def

# Set the logging file
logging = get_logger(__name__)


def run_from_container(image: str, command: str, binds: Dict = {}, container_type='docker', verbose: bool = True):
    """
    Runs a command inside a docker or singularity container and returns only when container has exited 
    (on success or fail)
    @param image: name or path to image to use
    @param command: command to pass to run/exec
    @param binds: (dict) A dictionary to configure volumes mounted inside the container. 
        The key is either the host path or a volume name, and the value is a container path.
        Currently hardcoded to mount the volumes read/write
    @param container_type: specifies whether to use "docker" or "singularity"
    @param verbose: if True, will print logs as container if executed
    @return:
    """
    stream = True if verbose else False
    if container_type == 'docker':
        binds = {k: {'bind': v, 'mode': 'rw'} for k, v in binds.items()}
        client = docker.from_env()
        qgis_pp_docker_img = client.images.pull(image)
        container = client.containers.run(
            image=qgis_pp_docker_img,
            command=command,
            volumes=binds,
            detach=True)
        exit_code = container.wait(timeout=1200)
        logging.info(exit_code)
        logs = container.logs(stream=stream)
        if stream:
            for line in logs:
                logging.info(codecs.decode(line))
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
        logging.info(f"\nContainer type is not valid. Choose 'docker' or 'singularity'")


def ras2vec(raster_file, output_path):
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
    for num_layer in num_layers:
        polygons = [feature for feature in out_features if feature['properties']['value'] == num_layer]
        layer_name = 'vector_' + str(num_layer).rjust(3, '0')
        logging.info("   - Writing layer: {}".format(layer_name))

        with fiona.open(output_path, 'w',
                        crs=to_string(src.crs),
                        layer=layer_name,
                        schema=feat_schema,
                        driver='GPKG') as dest:
            for polygon in polygons:
                dest.write(polygon)
    logging.info("\nNumber of features written: {}".format(i))


def main(params):
    """High-level postprocess pipeline.
    Runs polygonization and simplificaiton of a raster prediction and saves results to gpkg file.
    Args:
        params: configuration parameters
    """
    # Main params
    item_url = get_key_def('input_stac_item', params['inference'], expected_type=str, validate_path_exists=True)
    root = get_key_def('root_dir', params['inference'], default="data", to_path=True, validate_path_exists=True)

    # Postprocessing
    polygonization = get_key_def('polygonization', params['inference'], expected_type=bool, default=True)
    confidence_values = get_key_def('confidence_values', params['inference'], expected_type=bool, default=True)
    generalization = get_key_def('generalization', params['inference'], expected_type=bool, default=True)
    docker_img = get_key_def('docker_img', params['inference'], expected_type=str)
    singularity_img = get_key_def('singularity_img', params['inference'], expected_type=str,
                                  validate_path_exists=True)
    image = docker_img if docker_img else singularity_img if singularity_img else None
    container_type = 'docker' if docker_img else 'singularity' if singularity_img else None
    qgis_models_dir = get_key_def('qgis_models_dir', params['inference'], expected_type=str, validate_path_exists=True)

    dm = InferenceDataModule(root_dir=root,
                             item_path=item_url,
                             outpath=root/f"{Path(item_url).stem}_pred.tif",
                             )
    dm.setup()

    outpath = Path(dm.inference_dataset.outpath)
    outpath_heat = Path(dm.inference_dataset.outpath_heat)
    out_vect = Path(dm.inference_dataset.outpath_vec)

    # Postprocess final raster prediction (polygonization)
    if polygonization:
        out_vect_temp = root / f"{out_vect.stem}_temp.gpkg"
        out_vect_raw = root / f"{out_vect.stem}_raw.gpkg"
        logging.info(f"Polygonizing prediction to {out_vect}...")
        commands = [f"qgis_process plugins enable grassprovider",
                    f"qgis_process run grass7:r.to.vect -- input=/home/{str(outpath.name)} type=2 "
                    f"output=/home/{str(out_vect_temp.name)}",
                    f"qgis_process run native:extractbyattribute -- INPUT=/home/{str(out_vect_temp.name)} "
                    f"FIELD=value OPERATOR=2 VALUE=0 OUTPUT=/home/{str(out_vect_raw.name)}"]
        for command in commands:
            logging.debug(command)
            try:  # will raise Exception if image is None --> default to ras2vec
                run_from_container(image=image, command=command,
                                   binds={f"{str(outpath.parent.absolute())}": "/home"},
                                   container_type=container_type)
            except Exception as e:
                logging.error(f"\nError polygonizing using {container_type} container with image {image}."
                              f"\nCommand: {command}"
                              f"\nError {type(e)}: {e}")
                ras2vec(outpath, out_vect_raw)

        # in some cases, Grass fails to read source epsg and writes output without crs
        with rasterio.open(outpath) as src:
            gdf = geopandas.read_file(out_vect_raw)
            if gdf.crs != src.crs:
                gdf.set_crs(crs=src.crs, allow_override=True)
                # write file
                out_vect_raw_crs = out_vect_raw.parent / f"{out_vect_raw.stem}_crs.gpkg"
                gdf.to_file(out_vect_raw_crs)

        if out_vect_raw.is_file():
            logging.info(f'\nPolygonization completed. Raw prediction: {out_vect_raw}')

        if confidence_values and outpath_heat.is_file():
            with rasterio.open(outpath_heat, 'r') as src:
                gdf = geopandas.read_file(out_vect_raw)
                gdf['confidence'] = None
                for row, bbox in zip(gdf.iterrows(), gdf.envelope):
                    index, feature = row
                    left, bottom, right, top = bbox.bounds
                    window = rasterio.windows.from_bounds(left, bottom, right, top, transform=src.transform)
                    conf_vals = src.read(window=window)  # only read band relative to feature class
                    mask = conf_vals.argmax(axis=0) == feature.value
                    confidence = conf_vals[feature.value][mask].mean()
                    gdf.iloc(index)
                    feature.confidence = round(confidence.astype(int))  # FIXME not writing to GDF!
                gdf.to_file('test.gpkg')
        elif confidence_values:
            logging.error(f"Cannot add confidence levels to polygons. A heatmap must be generated at inference")

        if out_vect_raw.is_file() and generalization:  # generalize only if polygonization is successful
            logging.info(f"Generalizing prediction to {out_vect}")
            commands = [f"qgis_process plugins enable geo_sim_processing",
                        f"qgis_process plugins enable processing",
                        f"qgis_process run /models/1classe_cleanup_building_v2.model3 -- "
                        f"building=/home/{str(out_vect_raw.name)} "
                        f"Geopackagename=/home/{str(out_vect.name)} "
                        f"NomdelacoucheBatimentdanslegpkg=building Simplify=0.3 Deletehole=40"]  # FIXME: convert to degrees if crs 4326
            for command in commands:
                logging.debug(command)
                try:  # will raise Exception if image is None --> default to ras2vec
                    run_from_container(image=image, command=command,
                                       binds={f"{str(outpath.parent.absolute())}": "/home",
                                              f"{str(qgis_models_dir)}": "/models"},
                                       container_type=container_type)
                except Exception as e:
                    logging.error(f"\nError generalizing using {container_type} container with image {image}."
                                  f"\nCommand: {command}"
                                  f"\nError {type(e)}: {e}")
            if out_vect.is_file():
                logging.info(f'\nGeneralization completed. Final prediction: {out_vect}')
            else:
                logging.error(f'\nGeneralization failed. See logs...')

    logging.info(f'\nEnd of postprocessing')
