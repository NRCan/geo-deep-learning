# Licensed under the MIT License.
# Adapted from: https://github.com/microsoft/torchgeo/blob/3f7e525fbd01dddd25804e7a1b7634269ead1760/evaluate.py

# Hardware requirements: 64 Gb RAM (cpu), 8 Gb GPU RAM.

"""CCMEO postprocess script."""
import logging
import subprocess
from pathlib import Path

import docker
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
    docker_img = get_key_def('docker_img', params['inference'], default='remtav/qgis_pp:latest', expected_type=str)
    singularity_img = get_key_def('singularity_img', params['inference'], default=None, expected_type=str,
                                  validate_path_exists=True)
    qgis_models_dir = get_key_def('qgis_models_dir', params['inference'],
                                  default="/home/user/.local/share/QGIS/QGIS3/profiles/default/processing/models",
                                  expected_type=str, validate_path_exists=True)

    dm = InferenceDataModule(root_dir=root,
                             item_path=item_url,
                             outpath=root/f"{Path(item_url).stem}_pred.tif",
                             )
    dm.setup()

    outpath = Path(dm.inference_dataset.outpath)
    out_vect = Path(dm.inference_dataset.outpath_vec)

    # Postprocess final raster prediction (polygonization)
    if polygonization:
        out_vect_temp = root / f"{out_vect.stem}_temp.gpkg"
        logging.info(f"Vectorizing prediction to {out_vect}")
        commands = [f"qgis_process plugins enable grassprovider",
                    f"qgis_process run grass7:r.to.vect -- input=/home/{str(outpath.name)} type=2 "
                    f"output=/home/{str(out_vect_temp.name)}",
                    f"ogr2ogr \"/home/{str(out_vect.name)}\" \"/home/{str(out_vect_temp.name)}\" -where \"value\" > 0"]  # FIXME
        for command in commands:
            print(command)
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
                    container.logs(stream=True)  # FIXME: print logs!
                    continue
                except DockerException as e:
                    logging.info(f"Cannot postprocess using Docker: {e}\n")
            # Singularity: validate installation and assert version >= 3.0.0
            if singularity_img and Client.version() and int(Client.version().split(' ')[-1].split('.')[0]) >= 3:
                print(command.split(" "))
                try:
                    executor = Client.execute(to_absolute_path(str(singularity_img)), command.split(" "),
                                              bind=f"{str(outpath.parent.absolute())}:/home", stream=True)
                    for line in executor:
                        print(line)
                except subprocess.CalledProcessError as e:
                    logging.info(e)
                continue
            else:
                logging.info(f"Cannot postprocess using Singularity, will resort to less efficient vectorization"
                             f"with rasterio and shapely\n")
            ras2vec(outpath, out_vect)

        if generalization:
            out_vect_temp = root / f"{out_vect.stem}_temp.gpkg"
            logging.info(f"Simplifying prediction to {out_vect}")
            command = [f"qgis_process run model:cleanup_building -- building=/home/{str(out_vect_temp.name)} "
                        f"Geopackagename=/home/{str(out_vect.name)} "
                        f"NomdelacoucheBatimentdanslegpkg=building Simplify=0.3 Deletehole=40"]
            print(command)
            if docker_img:
                try:
                    client = docker.from_env()
                    qgis_pp_docker_img = client.images.pull(docker_img)
                    container = client.containers.run(
                        image=qgis_pp_docker_img,
                        command=command,
                        volumes={f'{str(outpath.parent.absolute())}': {'bind': '/home', 'mode': 'rw'},
                                 f'{str(qgis_models_dir)}':
                                     {'bind': '/usr/share/qgis/profiles/default/processing/models', 'mode': 'rw'}},
                        detach=True)
                    exit_code = container.wait(timeout=1200)
                    logging.info(exit_code)
                    container.logs(stream=True)  # FIXME: print logs!
                except DockerException as e:
                    logging.info(f"Cannot postprocess using Docker: {e}\n")
            if singularity_img:
                pass

    logging.info(f'\nPostprocessing completed on {dm.inference_dataset.item_url}'
                 f'\nFinal prediction written to {dm.inference_dataset.outpath_vec}')
