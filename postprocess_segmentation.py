# Licensed under the MIT License.
# Authors: Victor Alhassan, Rémi Tavon

# Hardware requirements: 64 Gb RAM (cpu), 8 Gb GPU RAM.

"""CCMEO postprocess script."""
import codecs
import os
import re
import shutil
import subprocess
from typing import Dict, Union
from collections import OrderedDict
from pathlib import Path

import docker
import fiona
import geopandas
import omegaconf
import rasterio
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from pandas.io.common import is_url
from rasterio import features
from rasterio.plot import reshape_as_image
from rasterio.windows import Window
from torch.hub import load_state_dict_from_url, get_dir
from tqdm import tqdm

from utils.logger import get_logger
from utils.utils import get_key_def, override_model_params_from_checkpoint, checkpoint_converter, read_checkpoint, \
    extension_remover, class_from_heatmap, ckpt_is_compatible

# Set the logging file
logging = get_logger(__name__)


def regularize_buildings(in_pred: Union[str, Path],
                         out_pred: Union[str, Path],
                         container_image: str,
                         container_type: str,
                         container_command: str,
                         code_dir: str,
                         building_value: int = 1,
                         fallback: bool = True,
                         fallback_models_dir: Union[str, Path] = "saved_models_gan"):
    """
    Regularizes building segmentation masks, i.e. uses a generative adversarial network (GAN) to perform a
    regularization of building boundaries to make them more realistic, i.e., having more rectilinear outlines
    which construct right angles if required.
    @param in_pred:
        input segmentation mask as raster
    @param out_pred:
        output regularized mask (as raster)
    @param container_image:
        docker/singularity image to create a container from
    @param container_type:
        docker or singularity
    @param container_command:
        command to pass to container for regularization
    @param code_dir: str
        Source directory where regularization code can be found. Will be bound to /media directory in container.
    @param building_value:
        pixel value of building predictions in segmentation mask
    @param fallback:
        if True, will resort to direct Python regularization if container approach fails
    @param fallback_models_dir:
        directory containing pretrained models if regularization falls back to direct Python

    @return:
    """
    try:  # will raise Exception if image is None --> default to ras2vec
        run_from_container(image=container_image, command=container_command,
                           binds={f"{str(in_pred.parent.absolute())}": "/home",
                                  f"{code_dir}": "/media"},
                           container_type=container_type)
        logging.info(f'\nRegularization completed')
    except Exception as e:
        logging.error(f"\nError regularizing using {container_type} container with image {container_image}."
                      f"\ncommand: {container_command}"
                      f"\nError {type(e)}: {e}"
                      f"\nWill try regularizing without container...")
        if fallback:
            try:
                from regularization import regularize
                if not fallback_models_dir:
                    fallback_models_dir = Path(get_dir()) / 'checkpoints'
                    ckpts = ["https://github.com/remtav/projectRegularization/blob/light/regularization/saved_models_gan/E140000_net?raw=true",
                              "https://github.com/remtav/projectRegularization/blob/light/regularization/saved_models_gan/E140000_e1?raw=true"]
                    for ckpt in ckpts:
                        load_state_dict_from_url(url=ckpt, model_dir=fallback_models_dir)
                regularize.main(
                    in_raster=in_pred,
                    out_raster=out_pred,
                    build_val=building_value,
                    models_dir=fallback_models_dir,
                )
            except ImportError:
                logging.critical(f"Failed to regularize buildings")


def polygonize(in_raster: Union[str, Path],
               out_vector: Union[str, Path],
               container_image: str = None, 
               container_type: str = 'docker', 
               container_command: str = '',
               fallback: bool = True):
    """
    Polygonizes a segmentation mask to GeoPackage
    @param in_raster:
        Path to input segmentation mask as raster (used only if fallback)
    @param out_vector:
        Path to output segmentation mask as raster (used only if fallback)
    @param container_image:
        docker/singularity image to create a container from
    @param container_type:
        docker or singularity
    @param container_command:
        command to pass to container for polygonization
    @param fallback:
        if True, will resort to direct Python polygonization if container approach fails
    @return:
    """
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
            logging.critical(f"\nOutputted polygonized prediction may be empty.")
            raise e
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
        if exit_code['StatusCode'] != 0:
            raise IOError(f"Error while executing docker with subprocess. Return code: {exit_code}")
    # Singularity: validate installation and assert version >= 3.0.0
    elif container_type == 'singularity':
        # Work around to prevent string parsing error: unexpected EOF while looking for matching `"'
        cmd_file = Path(to_absolute_path("command.sh"))
        with open(cmd_file, 'w') as dest:
            if "/bin/bash -c" not in command:
                logging.warning(
                    f"Singularity commands should start with \"/bin/bash -c\". Execution of container may fail")
            inner_cmd = command.split("/bin/bash -c")[-1].strip("\" ").strip("\"").replace("; ", "\n").replace(";",
                                                                                                               "\n")
            dest.write(inner_cmd)
        binds[f"{cmd_file.parent}"] = f"{cmd_file.parent}"
        command = "/bin/bash " + to_absolute_path("command.sh")
        binds = [f"--bind {k}:{v} " for k, v in binds.items()]
        binds_str = " "
        binds_str = binds_str.join(binds)
        command = f"singularity exec --nv --cleanenv {binds_str}{to_absolute_path(str(image))} {command}"
        logging.debug(command.split())
        subproc = subprocess.run(command.split())
        subproc = subproc.returncode
        if subproc != 0:
            raise IOError(f"Error while executing singularity with subprocess. Return code: {subproc}")
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


def add_confidence_from_heatmap(in_heatmap: Union[str, Path], in_vect: Union[str, Path], heatmap_threshold: int = 50):
    in_vect_copy = in_vect.parent / f"{in_vect.stem}_no_confid.gpkg"
    shutil.copy(in_vect, in_vect_copy)
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
    @param heatmap_threshold: 
        threshold (%) to apply to heatmap if single class prediction 
    @return:
    """
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
            conf_vals = reshape_as_image(conf_vals)
            flattened = class_from_heatmap(
                heatmap_arr=conf_vals,
                heatmap_threshold=heatmap_threshold,
                range_warning=False)
            mask = flattened == feature.value
            conf_vals = conf_vals[..., feature.value] if conf_vals.shape[-1] > 1 else conf_vals[..., 0]
            confidence = conf_vals[mask].mean()
            confidence = round(confidence.astype(int)) if confidence >= 0 else None
            confidences.append(confidence)
        gdf['confidence'] = confidences
        gdf.to_file(in_vect)
        os.remove(in_vect_copy)
        logging.info(f'\nConfidence values added to polygonized prediction. Output: {in_vect}')


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
    inf_outpath_ori = inf_outpath
    heatmap_name = get_key_def('heatmap_name', params['inference'], default=f"{inf_outpath.stem}_heatmap", expected_type=str)
    heatmap_name = extension_remover(heatmap_name)
    in_heatmap = root / f"{heatmap_name}.tif"
    outname = get_key_def('output_name', params['postprocess'], default=inf_outname, expected_type=str)
    outname = extension_remover(outname)
    if not inf_outpath.is_file():
        raise FileNotFoundError(f"\nCannot find raster prediction file to use for postprocessing."
                                f"\nGot:{inf_outpath}")
    logging.debug(f"\nPostprocessing root directory:\n{root}"
                  f"\ninput:\n{inf_outname}"
                  f"\noutput:\n{outname}")
    checkpoint = get_key_def('state_dict_path', params['postprocess'], expected_type=str, to_path=True,
                             validate_path_exists=True)
    dataset_classes_dict = get_key_def('classes_dict', params['dataset'], expected_type=DictConfig)

    # Post-processing
    confidence_values = get_key_def('confidence_values', params['postprocess'], expected_type=bool, default=True)
    heatmap_threshold = get_key_def('heatmap_threshold', params['inference'], default=50, expected_type=int)
    regularization = get_key_def('regularization', params['postprocess'], expected_type=bool, default=True)
    generalization = get_key_def('generalization', params['postprocess'], expected_type=bool, default=True)

    # output suffixes
    out_reg_suffix = get_key_def('regularization', params['postprocess']['output_suffixes'], default='_reg',
                                  expected_type=str)
    out_poly_suffix = get_key_def('polygonization', params['postprocess']['output_suffixes'], default='_raw',
                                  expected_type=str)
    out_gen_suffix = get_key_def('generalization', params['postprocess']['output_suffixes'], default='_post',
                                  expected_type=str)

    cont_type = get_key_def('cont_type', params['postprocess'], expected_type=str)
    # regularization container parameters
    reg_fallback = get_key_def('fallback', params['postprocess']['reg_cont'], expected_type=bool, default=True)
    reg_cont_image = get_key_def('cont_image', params['postprocess']['reg_cont'], expected_type=str)
    reg_code_dir = get_key_def('code_dir', params['postprocess']['reg_cont'], default=None, expected_type=str)
    try:
        reg_command = get_key_def('command', params['postprocess']['reg_cont'], expected_type=str)
    except omegaconf.errors.InterpolationKeyError:
        reg_command = None
    reg_models_dir = get_key_def('fallback_models_dir', params['postprocess']['reg_cont'], expected_type=str,
                                 to_path=True, validate_path_exists=True)
    
    # polygonization container parameters
    poly_fallback = get_key_def('fallback', params['postprocess']['poly_cont'], expected_type=bool, default=True)
    poly_cont_image = get_key_def('cont_image', params['postprocess']['poly_cont'], expected_type=str)
    poly_command = get_key_def('command', params['postprocess']['poly_cont'], expected_type=str)

    # generalization container parameters
    qgis_models_dir = get_key_def('qgis_models_dir', params['postprocess']['gen_cont'], expected_type=str,
                                  to_path=True, validate_path_exists=True)
    gen_cont_image = get_key_def('cont_image', params['postprocess']['gen_cont'], expected_type=str)
    gen_commands = dict(get_key_def('command', params['postprocess']['gen_cont'], expected_type=DictConfig))

    # fetch the footprint
    item_url = get_key_def('input_stac_item', params['inference'], expected_type=str, to_path=True, validate_path_exists=True)
    data_dir = get_key_def('raw_data_dir', params['dataset'], default="data", to_path=True, validate_path_exists=True)
    # TODO: Maybe add the download from the stac item if not there, but supposed since downloaded in inference
    footprint = os.path.join(data_dir, f'{os.path.basename(item_url)}-FOOTPRINT.geojson')
    if not os.path.isfile(footprint):
        logging.critical(f"\nFOOTPRINT not found! {footprint} is not a file.")
        footprint = None

    logging.debug('\nCreate "hparams" yaml to use pytorch lightning model management')
    logging.info(f"\nConverting geo-deep-learning checkpoint to pytorch lightning...")
    if is_url(checkpoint):
        load_state_dict_from_url(url=checkpoint, map_location='cpu', model_dir=models_dir)
        checkpoint = models_dir / Path(checkpoint).name
    if not ckpt_is_compatible(checkpoint):
        checkpoint = checkpoint_converter(in_pth_path=checkpoint, out_dir=models_dir)
    checkpoint_dict = read_checkpoint(checkpoint, out_dir=models_dir, update=False)
    params = override_model_params_from_checkpoint(params=params, checkpoint_params=checkpoint_dict["hyper_parameters"])

    logging.debug(f"\nFilter generalization commands based on extracted classes")
    classes_dict = get_key_def('classes_dict', params['dataset'], expected_type=DictConfig)
    classes_dict = {k: v for k, v in classes_dict.items() if v}  # Discard keys where value is None
    logging.debug(f"\nGeneralization commands, before filtering for extracted classes:\n{gen_commands}")
    gen_cmds_pruned = {data_class: cmd for data_class, cmd in gen_commands.items() if data_class in classes_dict.keys()}
    logging.debug(f"\nGeneralization commands, after filtering for extracted classes:\n{gen_cmds_pruned}")
    # WARNING: this is highly coupled to values in generalization commands in config.
    if 'BUIL' in classes_dict.keys():
        reg_command = reg_command.replace(f"--build-val {dataset_classes_dict['BUIL']}",
                                          f"--build-val {classes_dict['BUIL']}")
        logging.debug(f"\nRegularization command:\n{reg_command}")
    logging.debug(f"\nAttribute values are overridden according to output class values contained in model checkpoint\n")
    gen_cmds_pruned = {cls: cmd.replace(f"--inselectattrint={dataset_classes_dict[cls]}",
                                        f"--inselectattrint={classes_dict[cls]}")
                       for cls, cmd in gen_cmds_pruned.items()}
    logging.debug(f"\nGeneralization commands, after override of class values:\n{gen_cmds_pruned}")

    # build output paths
    out_reg = root / f"{inf_outname}{out_reg_suffix}.tif"
    out_poly = root / f"{inf_outname}{out_poly_suffix}.gpkg"
    out_gen = root / f"{outname}{out_gen_suffix}.gpkg"

    returned_vector_pred = None

    # TODO: run regularization after polygonization? if so, polygonized output needs to rasterized and polygonized again
    if 'BUIL' in classes_dict and regularization:
        logging.info(f'Regularizing prediction. Polygonization\'s input ("{inf_outpath}") will become "{out_reg}".'
                     f'\nRegularization output: {out_reg}')
        building_value = classes_dict['BUIL']
        regularize_buildings(
            in_pred=inf_outpath,
            out_pred=out_reg,
            container_image=reg_cont_image,
            container_type=cont_type,
            container_command=reg_command,
            code_dir=reg_code_dir,
            building_value=building_value,
            fallback=reg_fallback,
            fallback_models_dir=reg_models_dir,
        )
        if out_reg.is_file():
            if f"{inf_outname}.tif" not in poly_command:
                logging.critical(f"\nInput to polygonization command should be replaced by regularization output"
                                 f"\n\"{inf_outname}.tif\" not found in original command"
                                 f"\nFailed to replace \"{inf_outname}.tif\" with \"{out_reg.stem}.tif\" in command.")
            else:
                poly_command = poly_command.replace(f"{inf_outname}.tif", f"{out_reg.stem}.tif")
            inf_outpath = inf_outpath.parent / f"{out_reg.stem}.tif"
        else:
            logging.error(f'\nFailed to create regularized output "{out_reg}".'
                          f'\nPolygonization and generalization will continue on non-regularized inference.')

    # Postprocess final raster prediction (polygonization)
    try:
        polygonize(
            in_raster=inf_outpath,
            out_vector=out_poly,
            container_image=poly_cont_image,
            container_type=cont_type,
            container_command=poly_command,
            fallback=poly_fallback,
        )
        returned_vector_pred = out_poly
    except fiona.errors.DriverError as e:
        logging.critical(
            f"\n{type(e)}: {e}"
            f"\nSkipping all remaining postprocessing and exiting...")
        # FIXME: should be reverted to raising error once pretrained models are public and can be used in CI
        return

    # set confidence values to features in polygonized prediction
    if confidence_values and in_heatmap.is_file():
        add_confidence_from_heatmap(in_heatmap=in_heatmap, in_vect=out_poly, heatmap_threshold=heatmap_threshold)
    elif confidence_values:
        logging.error(f"Cannot add confidence levels to polygons. A heatmap must be generated at inference")

    # Clip the predicted polygon(s) if given one
    if footprint:
        logging.info(f'\nClipping predicted polygon(s). Footprint: {footprint}')
        
        gdf_poly = geopandas.read_file(returned_vector_pred)
        if gdf_poly.empty:
            logging.critical(f'\nThe raw prediction contain no polygon.')
        else:
            gdf_footprint = geopandas.read_file(footprint)
            gdf_footprint = gdf_footprint.to_crs(str(gdf_poly.crs))            
            gdf_clipped = geopandas.clip(gdf_poly, gdf_footprint)
            gdf_clipped.to_file(returned_vector_pred, driver="GPKG")
        
        logging.info(f'\nClipping completed. Clipped prediction: {returned_vector_pred}')

    if generalization:
        logging.info(f"Generalizing prediction to {out_gen}")
        for command in gen_cmds_pruned.values():
            try:  # will raise Exception if image is None --> default to ras2vec
                run_from_container(image=gen_cont_image, command=command,
                                   binds={f"{str(inf_outpath.parent.absolute())}": "/home",
                                          f"{str(qgis_models_dir)}": "/models"},
                                   container_type=cont_type)
            except Exception as e:
                logging.error(f"\nError generalizing using {cont_type} container with image {gen_cont_image}."
                              f"\ncommand: {command}"
                              f"\nError {type(e)}: {e}")
            # Collect the name of the layer created
            if 'ROAI' in classes_dict: # take the polygon layer for the road
                m = re.search('outlayernamepoly=(.{6})', str(command))
            else:
                m = re.search('outlayername=(.{6})', str(command))
            layer_name = m.group(1)
            
        if out_gen.is_file():
            logging.info(f'\nGeneralization completed. Final prediction: {out_gen}')
            returned_vector_pred = out_gen
            
            try: # Try to convert multipolygon to polygon
                df = geopandas.read_file(returned_vector_pred, layer=layer_name)
                if 'MultiPolygon' in df['geometry'].geom_type.values:
                    logging.info("\nConverting multiPolygon to Polygon...")
                    gdf_exploded = df.explode(index_parts=True, ignore_index=True)
                    gdf_exploded.to_file(returned_vector_pred, layer=layer_name)
            except Exception as e:
                logging.error(f"\nSomething went wrong during the convertion of Polygon. \nError {type(e)}: {e}")
            
            # Fetch the tag information inside the tiff
            tiff_src = rasterio.open(inf_outpath_ori)
            tags = tiff_src.tags()
            # check if the tiff have a checkpoint save in 
            if 'checkpoint' in tags.keys():
                checkpoint_path = tags['checkpoint']
                # add more info on a new layer waiting for Fiona 1.9
                logging.info(f"\nAdding a layer 'extent_info' in {out_gen} with addintional informations.")
                # Create a schema to store extent informations in the gpks
                extent_schema_list = [("stac_item", 'str'), ("gdl_image", 'str')]
                # list all layer(s) in the gpkg
                layers = fiona.listlayers(out_gen)
                # Look which class is use first by looking in the inf_outpath 
                cls_name = next((elem for elem in classes_dict if elem in str(inf_outpath_ori)), None)  # TODO something when its None
                # Check if the layer already created
                if 'extent_info' in layers:
                    # open the gpkg
                    _gpkg = fiona.open(out_gen, layer='extent_info')
                    new_extent_schema = _gpkg.schema.copy()
                    # Add the model class name to the schema to fit what will be added
                    new_extent_schema['properties'].update({"model_"+cls_name : 'str'})
                    # extract information in the layer 
                    prop = list(_gpkg.filter())[0]['properties']
                    p = prop.copy()
                    # close the gpkg to be able to rewrite it
                    _gpkg.close()
                    # add infrmation in the layer already there
                    with fiona.open(returned_vector_pred, 'w', crs=_gpkg.crs, layer='extent_info', schema=new_extent_schema, driver='GPKG', overwrite=True) as ds:
                        # add the checkpoint path associated with the model use for that class
                        p.update({"model_"+cls_name : checkpoint_path})
                        ds.write({'properties': p})
                # If not there create the layer for extent information
                else:
                    # open the gpkg
                    _gpkg = fiona.open(out_gen)
                    # Add the model class name to the schema to fit what will be added
                    extent_schema_list.append(("model_"+cls_name, 'str'))
                    extent_schema = {'properties': OrderedDict(extent_schema_list)}
                    # Create a new layer in the gpkg with all information on stac item, gdl image and model class
                    with fiona.open(returned_vector_pred, 'w', crs=_gpkg.crs, layer='extent_info', schema=extent_schema, driver='GPKG') as ds:
                        p = {
                            "stac_item": item_url,
                            "gdl_image": reg_cont_image,
                        }
                        # add the checkpoint path associated with the model use for that class
                        p.update({"model_"+cls_name : checkpoint_path})
                        ds.write({'properties': p})
            
        else:
            logging.error(f'\nGeneralization failed. Output "{out_gen}" not created. See logs...')
        
    logging.info(f'\nEnd of postprocessing')

    return returned_vector_pred
