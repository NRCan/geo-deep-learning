# Licensed under the MIT License.
# Adapted from: https://github.com/microsoft/torchgeo/blob/3f7e525fbd01dddd25804e7a1b7634269ead1760/evaluate.py

# Hardware requirements: 64 Gb RAM (cpu), 8 Gb GPU RAM.

"""CCMEO postprocess script."""
import codecs
import subprocess
from pathlib import Path

import docker
import geopandas
import rasterio
from docker.errors import DockerException
from hydra.utils import to_absolute_path
from spython.main import Client

from inference.InferenceDataModule import InferenceDataModule
from inferencedev_segmentation import ras2vec
from utils.logger import get_logger
from utils.utils import get_key_def

# Set the logging file
logging = get_logger(__name__)


def main(params):
    """High-level postprocess pipeline.
    Runs polygonization and simplificaiton of a raster prediction and saves results to gpkg file.
    Args:
        params: configuration parameters
    """
    # Main params
    item_url = get_key_def('input_stac_item', params['inference'], expected_type=str)  #, to_path=True, validate_path_exists=True) TODO implement for url
    root = get_key_def('root_dir', params['inference'], default="data", to_path=True, validate_path_exists=True)

    # Postprocessing
    polygonization = get_key_def('polygonization', params['inference'], expected_type=bool, default=True)
    generalization = get_key_def('generalization', params['inference'], expected_type=bool, default=True)
    docker_img = get_key_def('docker_img', params['inference'], expected_type=str)
    singularity_img = get_key_def('singularity_img', params['inference'], default=None, expected_type=str,
                                  validate_path_exists=True)
    qgis_models_dir = get_key_def('qgis_models_dir', params['inference'], expected_type=str, validate_path_exists=True)

    dm = InferenceDataModule(root_dir=root,
                             item_path=item_url,
                             outpath=root/f"{Path(item_url).stem}_pred.tif",
                             )
    dm.setup()

    outpath = Path(dm.inference_dataset.outpath)
    out_vect = Path(dm.inference_dataset.outpath_vec)

    # Postprocess final raster prediction (polygonization) TODO: refactor docker/singularity calls
    if polygonization:
        out_vect_temp = root / f"{out_vect.stem}_temp.gpkg"
        out_vect_raw = root / f"{out_vect.stem}_raw.gpkg"
        logging.info(f"Vectorizing prediction to {out_vect}")
        commands = [f"qgis_process plugins enable grassprovider",
                    f"qgis_process run grass7:r.to.vect -- input=/home/{str(outpath.name)} type=2 "
                    f"output=/home/{str(out_vect_temp.name)}",
                    f"qgis_process run native:extractbyattribute -- INPUT=/home/{str(out_vect_temp.name)} "
                    f"FIELD=value OPERATOR=2 VALUE=0 OUTPUT=/home/{str(out_vect_raw.name)}"]
        for command in commands:
            logging.debug(command)
            # Docker
            if docker_img:
                try:
                    client = docker.from_env()
                    qgis_pp_docker_img = client.images.pull(docker_img)
                    container = client.containers.run(
                        image=qgis_pp_docker_img,
                        command=command,
                        volumes={f'{str(outpath.parent.absolute())}': {'bind': '/home', 'mode': 'rw'}},
                        detach=True)
                    exit_code = container.wait(timeout=1200)
                    logging.info(exit_code)
                    logs = container.logs(stream=True)
                    for line in logs:
                        print(codecs.decode(line))
                    continue
                except DockerException as e:
                    logging.info(f"Cannot postprocess using Docker: {e}\n")
            # Singularity: validate installation and assert version >= 3.0.0
            if singularity_img and Client.version() and int(Client.version().split(' ')[-1].split('.')[0]) >= 3:
                logging.debug(command.split(" "))
                try:
                    executor = Client.execute(to_absolute_path(str(singularity_img)), command.split(" "),
                                              bind=f"{str(outpath.parent.absolute())}:/home", stream=True)
                    for line in executor:
                        logging.info(line)
                except subprocess.CalledProcessError as e:
                    logging.info(e)
                continue
            else:
                logging.info(f"\nCannot postprocess using Singularity, will resort to less efficient vectorization"
                             f"with rasterio and shapely")
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

        if out_vect_raw.is_file() and generalization:  # generalize only polygonization is successful
            logging.info(f"Generalizing prediction to {out_vect}")
            commands = [f"qgis_process plugins enable geo_sim_processing",
                        f"qgis_process plugins enable processing",
                        f"qgis_process run /models/1classe_cleanup_building_v2.model3 -- "
                        f"building=/home/{str(out_vect_raw.name)} "
                        f"Geopackagename=/home/{str(out_vect.name)} "
                        f"NomdelacoucheBatimentdanslegpkg=building Simplify=0.3 Deletehole=40"]
            for command in commands:
                logging.debug(command)
                if docker_img:
                    try:
                        client = docker.from_env()
                        qgis_pp_docker_img = client.images.pull(docker_img)
                        container = client.containers.run(
                            image=qgis_pp_docker_img,
                            command=command,
                            volumes={f'{str(outpath.parent.absolute())}': {'bind': '/home', 'mode': 'rw'},
                                     f'{str(qgis_models_dir)}':
                                         {'bind': '/models',
                                          'mode': 'rw'}},
                            detach=True)
                        exit_code = container.wait(timeout=1200)
                        logging.info(exit_code)
                        logs = container.logs(stream=True)
                        for line in logs:
                            print(codecs.decode(line))
                    except DockerException as e:
                        logging.info(f"Cannot postprocess using Docker: {e}\n")
                # Singularity: validate installation and assert version >= 3.0.0
                if singularity_img and Client.version() and int(Client.version().split(' ')[-1].split('.')[0]) >= 3:
                    logging.debug(command.split(" "))
                    try:
                        executor = Client.execute(to_absolute_path(str(singularity_img)), command.split(" "),
                                                  bind=f"{str(outpath.parent.absolute())}:/home", stream=True)
                        for line in executor:
                            logging.info(line)
                    except subprocess.CalledProcessError as e:
                        logging.info(e)
                    continue
                else:
                    logging.info(f"\nCannot postprocess using Singularity")
            if out_vect.is_file():
                logging.info(f'\nGeneralization completed. Final prediction: {out_vect}')

    logging.info(f'\nEnd of postprocessing')
