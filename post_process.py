print('Importing packages and modules...')
import argparse
import multiprocessing
import logging
import logging.config

import numpy as np
import torch
from skimage import measure
from skimage.transform import rescale
from torch import nn

from utils.verifications import validate_raster

np.random.seed(1234)  # Set random seed for reproducibility
import rasterio
import time

from pathlib import Path
from tqdm import tqdm

from utils.utils import get_key_def, get_git_hash, subprocess_cmd, map_wrapper
from utils.readers import read_parameters, read_gdl_csv

logging.getLogger(__name__)

print('Done')

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            # nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3, stride=1, padding=1),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            # nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3, stride=1, padding=1),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, num_residual_blocks=8, in_features=256):
        super(GeneratorResNet, self).__init__()

        out_features = in_features

        model = []

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        # model += [nn.ReflectionPad2d(2), nn.Conv2d(out_features, 2, 7), nn.Softmax()]
        model += [nn.Conv2d(out_features, 2, 7, stride=1, padding=3), nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, feature_map):
        x = self.model(feature_map)
        return x


class Encoder(nn.Module):
    def __init__(self, channels=3 + 2):
        super(Encoder, self).__init__()

        # Initial convolution block
        out_features = 64
        model = [
            nn.Conv2d(channels, out_features, 7, stride=1, padding=3),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride=2),
            ]
            in_features = out_features

        self.model = nn.Sequential(*model)

    def forward(self, arguments):
        x = torch.cat(arguments, dim=1)
        x = self.model(x)
        return x


def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def predict_building(rgb, mask, model):
    Tensor = torch.cuda.FloatTensor

    mask = to_categorical(mask, 2)

    rgb = rgb[np.newaxis, :, :, :]
    mask = mask[np.newaxis, :, :, :]

    E, G = model

    rgb = Tensor(rgb)
    mask = Tensor(mask)
    rgb = rgb.permute(0, 3, 1, 2)
    mask = mask.permute(0, 3, 1, 2)
    # print(rgb.shape)
    # print(mask.shape)

    rgb = rgb / 255.0

    # PREDICTION
    pred = G(E([rgb, mask]))
    pred = pred.permute(0, 2, 3, 1)

    pred = pred.detach().cpu().numpy()

    pred = np.argmax(pred[0, :, :, :], axis=-1)
    return pred


def fix_limits(i_min, i_max, j_min, j_max, min_image_size=256):
    def closest_divisible_size(size, factor=4):
        while size % factor:
            size += 1
        return size

    height = i_max - i_min
    width = j_max - j_min

    # pad the rows
    if height < min_image_size:
        diff = min_image_size - height
    else:
        diff = closest_divisible_size(height) - height + 16

    i_min -= (diff // 2)
    i_max += (diff // 2 + diff % 2)

    # pad the columns
    if width < min_image_size:
        diff = min_image_size - width
    else:
        diff = closest_divisible_size(width) - width + 16

    j_min -= (diff // 2)
    j_max += (diff // 2 + diff % 2)

    return i_min, i_max, j_min, j_max


def regularization(rgb, ins_segmentation, model, in_mode="instance", out_mode="instance", min_size=10):
    assert in_mode == "semantic"
    assert out_mode == "instance" or out_mode == "semantic"

    border = 256

    if rgb is None:
        logging.debug(ins_segmentation.shape)
        rgb = np.zeros((ins_segmentation.shape[0], ins_segmentation.shape[1], 3), dtype=np.uint8)

    print('Padding...')
    ins_segmentation = np.pad(array=ins_segmentation, pad_width=border, mode='constant', constant_values=0)
    npad = ((border, border), (border, border), (0, 0))
    rgb = np.pad(array=rgb, pad_width=npad, mode='constant', constant_values=0)
    # print(f'RGB l151: {rgb.shape}')
    print('Done')

    print('Counting buildings...')
    contours_list = measure.find_contours(ins_segmentation)
    # ins_segmentation = np.uint16(measure.label(ins_segmentation, background=0))

    max_instance = len(contours_list)
    # max_instance = np.amax(ins_segmentation)
    print(f'Found {max_instance} buildings!')

    regularization = np.zeros(ins_segmentation.shape, dtype=np.uint16)

    batch_size = 4
    for ins in tqdm(range(1, max_instance + 1), desc="Regularization"):
        # print(f'Computing indices for instance {ins}...')
        # indices = np.argwhere(ins_segmentation==ins)
        indices = contours_list[ins - 1]
        # print('Done')
        building_size = indices.shape[0]
        if building_size > min_size:
            i_min = int(np.amin(indices[:, 0]))
            i_max = int(np.amax(indices[:, 0]))
            j_min = int(np.amin(indices[:, 1]))
            j_max = int(np.amax(indices[:, 1]))

            # print(i_min, i_max, j_min, j_max)
            i_min, i_max, j_min, j_max = fix_limits(i_min, i_max, j_min, j_max)
            # print('Fixed limits:')
            # print(i_min, i_max, j_min, j_max)

            if (i_max - i_min) > 10000 or (j_max - j_min) > 10000:
                continue

            mask = np.copy(ins_segmentation[i_min:i_max, j_min:j_max] == 255)

            rgb_mask = np.copy(rgb[i_min:i_max, j_min:j_max, :])
            # print(f'RGB mask l168: {rgb_mask.shape}')

            max_building_size = 768
            rescaled = False
            if mask.shape[0] > max_building_size and mask.shape[0] >= mask.shape[1]:
                f = max_building_size / mask.shape[0]
                mask = rescale(mask, f, anti_aliasing=False, preserve_range=True)
                rgb_mask = rescale(rgb_mask, f, anti_aliasing=False, multichannel=True)
                rescaled = True
                # print(f'RGB mask l179: {rgb_mask.shape}')
            elif mask.shape[1] > max_building_size and mask.shape[1] >= mask.shape[0]:
                # print(f'mask shape: {mask.shape}')
                f = max_building_size / mask.shape[1]
                # print(f'f: {f}')
                mask = rescale(mask, f, anti_aliasing=False)
                rgb_mask = rescale(rgb_mask, f, anti_aliasing=False, preserve_range=True, multichannel=True)
                rescaled = True
                # print(f'RGB mask l185: {rgb_mask.shape}')

            # rint(f'RGB mask l187: {rgb_mask.shape}')
            pred = predict_building(rgb_mask, mask, model)

            if rescaled:
                pred = rescale(pred, 1 / f, anti_aliasing=False, preserve_range=True)

            pred_indices = np.argwhere(pred != 0)

            if pred_indices.shape[0] > 0:
                pred_indices[:, 0] = pred_indices[:, 0] + i_min
                pred_indices[:, 1] = pred_indices[:, 1] + j_min
                x, y = zip(*pred_indices)
                if out_mode == "semantic":
                    regularization[x, y] = 1
                else:
                    regularization[x, y] = ins

    return regularization[border:-border, border:-border]


def arr_threshold(arr, value=127):
    bool_M = (arr >= value)
    arr[bool_M] = 255
    arr[~bool_M] = 0
    # print(np.unique(M))
    return arr


def regularize_buildings(pred_arr, model_dir, sat_img_arr=None, apply_threshold=None):
    if apply_threshold:
        print('Applying threshold...')
        pred_arr = arr_threshold(pred_arr, value=apply_threshold)
    logging.debug(pred_arr.shape)

    print('Done')
    model_encoder = Path(model_dir) / "E140000_e1"
    model_generator = Path(model_dir) / "E140000_net"
    E1 = Encoder()
    G = GeneratorResNet()
    G.load_state_dict(torch.load(model_generator))
    E1.load_state_dict(torch.load(model_encoder))
    E1 = E1.cuda()
    G = G.cuda()

    model = [E1, G]
    R = regularization(sat_img_arr, pred_arr, model, in_mode="semantic", out_mode="semantic")
    return R


def post_process_pipeline(inference_raster, outdir, apply_threshold=False, buildings_model=None, simp_tolerance=0.2):
    is_valid_raster, _ = validate_raster(inference_raster)
    if not is_valid_raster:
        logging.error(f"{inference_raster} is not a valid raster")
        return

    if buildings_model is not None:
        logging.debug(buildings_model)
        with rasterio.open(inference_raster, 'r') as raw_pred:
            outname_reg = outdir / f'{inference_raster.stem}_reg.tif'
            if not outname_reg.is_file():
                logging.debug(f'Regularizing buildings in {inference_raster}...')
                meta = raw_pred.meta
                raw_pred_arr = raw_pred.read()[0, ...]
                # FIXME: softcode for multiclass inference
                raw_pred_arr_buildings = raw_pred_arr
                #logging.debug(raw_pred_arr.shape)
                #raw_pred_arr_buildings = np.where(raw_pred_arr == 4, 255, 0)
                #logging.debug(raw_pred_arr_buildings.shape)
                #logging.debug(np.bincount(raw_pred_arr_buildings.flatten()))
                reg_arr = regularize_buildings(raw_pred_arr_buildings, buildings_model, apply_threshold=apply_threshold)
                #out_arr = np.where(reg_arr == 255, 4, 0)
                #out_arr = raw_pred_arr[raw_pred_arr < 4]
                #out_arr = out_arr[np.newaxis, :, :]

                meta.update({"dtype": 'uint8', "compress": 'lzw'})
                with rasterio.open(outname_reg, 'w+', **meta) as reg_pred:
                    logging.info(f'Successfully regularized on {inference_raster}\nWriting to file: {outname_reg}')
                    reg_pred.write(reg_arr.astype(np.uint8) * 255)
            else:
                logging.info(f'Regularized prediction exists: {outname_reg}')

    outname_reg_vec_temp = outdir / f'{outname_reg.stem}_temp.gpkg'
    if not outname_reg_vec_temp.is_file():
        cmd = f"gdal_polygonize.py -8 {outname_reg} {outname_reg_vec_temp}"
        logging.debug(cmd)
        suc_msg = f'Successfully polygonized {outname_reg}\nOutput file: {outname_reg_vec_temp}'
        fail_msg = f'Failed to polygonized {outname_reg}\nCouldn\'t create output file: {outname_reg_vec_temp}'
        subprocess_cmd(cmd, suc_msg, fail_msg)
    else:
        logging.info(f'Polygonized raster exists: {outname_reg_vec_temp}')

    outname_reg_vec = outdir / f'{outname_reg.stem}_raw.gpkg'
    if not outname_reg_vec.is_file():
        cmd = f'ogr2ogr -progress {outname_reg_vec} {outname_reg_vec_temp} -where \"\\\"DN\\\" > 0\"'
        logging.debug(cmd)
        suc_msg = f'Successfully filtered features  {outname_reg}\nOutput file: {outname_reg_vec}'
        fail_msg = f'Polygonized raster exists: {outname_reg_vec}'
        subprocess_cmd(cmd, suc_msg, fail_msg, use_spcall=False)

    final_gpkg = outdir / f'{outname_reg_vec.stem}_simp.gpkg'
    if not final_gpkg.is_file():
        cmd = f'ogr2ogr -progress {final_gpkg} {outname_reg_vec_temp} -where \"\\\"DN\\\" > 0\"' \
              f' -simplify {simp_tolerance}'
        logging.debug(cmd)
        suc_msg = f'Successfully simplified {outname_reg_vec}\nOutput file: {final_gpkg}'
        fail_msg = f'Simplified vector inference exists: {final_gpkg}'
        subprocess_cmd(cmd, suc_msg, fail_msg, use_spcall=False)


def main(params):
    """
    -------
    :param params: (dict) Parameters found in the yaml config file.
    """
    start_time = time.time()

    # mlflow logging
    mlflow_uri = get_key_def('mlflow_uri', params['global'], default="./mlruns")
    exp_name = get_key_def('mlflow_experiment_name', params['global'], default='gdl-training', expected_type=str)

    # MANDATORY PARAMETERS
    default_csv_file = Path(get_key_def('preprocessing_path', params['global'], ''),
                            exp_name, f"inference_sem_seg_{exp_name}.csv")
    img_dir_or_csv = get_key_def('img_dir_or_csv_file', params['inference'], default_csv_file, expected_type=str)
    state_dict = get_key_def('state_dict_path', params['inference'], expected_type=str)
    num_classes = get_key_def('num_classes', params['global'], expected_type=int)
    num_bands = get_key_def('number_of_bands', params['global'], expected_type=int)

    # OPTIONAL PARAMETERS
    # basics
    debug = get_key_def('debug_mode', params['global'], False)
    parallel = get_key_def('parallelize', params['inference'], default=False, expected_type=bool)
    apply_threshold = get_key_def('apply_threshold', params['inference'], None, expected_type=int)
    simp_tolerance = get_key_def('simp_tolerance', params['inference'], 0.2, expected_type=float)
    # keeps this script relatively agnostic to source of inference data
    tiles_dir = get_key_def('tiles_dir', params['inference'], None, expected_type=str)
    if tiles_dir is not None and tiles_dir.is_dir():
        tiles_dir = Path(tiles_dir)
    else:
        raise NotADirectoryError(f"Couldn't locate tiles directory: {tiles_dir}")
    buildings_model = Path(get_key_def('buildings_reg_modeldir', params['inference'], None, expected_type=str))
    if buildings_model is not None and not buildings_model.is_dir():
        raise NotADirectoryError(f"Couldn't locate building regularization model directory: {buildings_model}")
    # FIXME: accept multiclass inference
    if num_classes > 1 and buildings_model is not None:
        raise ValueError(f'Value mismatch: when "buildings" is True, "num_classes" should be 1, not {num_classes}')

    # SETTING OUTPUT DIRECTORY
    working_folder = Path(state_dict).parent / f'inference_{num_bands}bands'
    if tiles_dir:
        Path.mkdir(working_folder, exist_ok=True)
    elif not working_folder.is_dir():
        raise NotADirectoryError("Couldn't find source inference directory")

    working_folder_pp = working_folder.parent / f'{working_folder.stem}_post-process'
    Path.mkdir(working_folder_pp, exist_ok=True)
    # add git hash from current commit to parameters if available. Parameters will be saved to hdf5s
    params['global']['git_hash'] = get_git_hash()

    # See: https://docs.python.org/2.4/lib/logging-config-fileformat.html
    log_config_path = Path('utils/logging.conf').absolute()
    console_level_logging = 'INFO' if not debug else 'DEBUG'
    log_file_prefix = 'post-process'
    logging.config.fileConfig(log_config_path, defaults={'logfilename': f'{working_folder_pp}/{log_file_prefix}.log',
                                                         'logfilename_error':
                                                             f'{working_folder_pp}/{log_file_prefix}_error.log',
                                                         'logfilename_debug':
                                                             f'{working_folder_pp}/{log_file_prefix}_debug.log',
                                                         'console_level': console_level_logging})

    if tiles_dir and Path(img_dir_or_csv).suffix == '.csv':
        inference_srcdata_list = read_gdl_csv(Path(img_dir_or_csv))
    elif tiles_dir and Path(img_dir_or_csv).is_dir():
        # TODO: test this. Only tested csv for now
        inference_srcdata_list = [{'tif': raster} for raster in Path(img_dir_or_csv).glob(f'**/*.tif')]
    elif tiles_dir:
        raise FileNotFoundError(f'Couldn\'t locate .csv file or directory "{img_dir_or_csv}" '
                                f'containing imagery for inference')

    # 1. TILING
    if tiles_dir:
        input_args = []
        logging.info(f"Merging tiles...")
        for info in inference_srcdata_list:
            is_valid_srcraster, _ = validate_raster(info['tif'])
            # post-process only if raster is valid and if 'dataset' column is empty or 'tst'
            if is_valid_srcraster and 'dataset' not in info.keys() or info['dataset'] == 'tst':
                inference_raster = working_folder / f"{Path(info['tif']).stem}_pred.tif"
                tiles_files = [str(tile) for tile in tiles_dir.glob(f"{Path(info['tif']).stem}_*.tif")]
                logging.info(f'{info["tif"]}: \n\tFound {len(tiles_files)} tiles to merge')
                tiles_files_str = " ".join(tiles_files)
                if not inference_raster.is_file():
                    cmd = f"gdal_merge.py -o {inference_raster} {tiles_files_str}"
                    logging.debug(cmd)
                    suc_msg = f'Successfully merged inference tiles from {Path(info["tif"])}\n' \
                              f'Merged raster: {inference_raster}'
                    fail_msg = f'Failed to merged inference tiles from {Path(info["tif"])}\n'\
                               f'Could\'t create merged raster: {inference_raster}'
                    # Have encountered problems with subprocess.run on Unix: "Arguments list too long". Subp.call ok.
                    if parallel:
                        input_args.append([subprocess_cmd, cmd, suc_msg, fail_msg])
                    else:
                        subprocess_cmd(cmd, suc_msg, fail_msg)
                else:
                    logging.info(f'Merged raster exists: {inference_raster}')
            else:
                logging.error(f"Failed to merge tiles from {info['tif']}")
                if not is_valid_srcraster:
                    logging.error(f"{info['tif']} is not a valid raster")
                else:
                    logging.error(f"If csv, only lines where 'dataset' column is 'tst' or empty is considered valid. "
                                  f"Check README.md for more information about expected csv file")

        if parallel:
            logging.info(f'Will merge tiles for {len(input_args)} inference images')
            with multiprocessing.Pool(None) as pool:
                pool.map(map_wrapper, input_args)

    inference_destdata_list = [raster for raster in Path(working_folder).glob(f'**/*.tif')]
    logging.info(f'\n\tFound {len(inference_destdata_list)} inference rasters to post-process\n'
                 f'\tCopying first entry:\n{inference_destdata_list[0]}\n')

    # 2. POLYGONIZATION PIPELINE
    input_args = []
    for inference_raster in tqdm(inference_destdata_list, position=0, leave=False):
        try:
            if parallel:
                input_args.append([post_process_pipeline, inference_raster, working_folder_pp, apply_threshold,
                                   buildings_model, simp_tolerance])
            else:
                # FIXME: one for loop for each PP step (complete all regularization first, then polygonize, etc.)
                post_process_pipeline(inference_raster, working_folder_pp, apply_threshold, buildings_model,
                                      simp_tolerance)
        except IOError as e:
            logging.error(f"Failed to post-process from {inference_raster}\n{e}")

    if parallel:
        logging.info(f'Will post-process {len(input_args)} inference images')
        # If regularizing buildings, multiprocessing is limited because of GPU operation. Fills up 16G GPU RAM if > 2
        num_threads = 2 if buildings_model else None
        with multiprocessing.get_context('spawn').Pool(num_threads) as pool:
            pool.map(map_wrapper, input_args)

    logging.info(f"End of process. Elapsed time: {int(time.time() - start_time)} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Buildings post processing')
    parser.add_argument('ParamFile', metavar='DIR',
                        help='Path to training parameters stored in yaml')
    args = parser.parse_args()
    params = read_parameters(args.ParamFile)
    print(f'\n\nStarting post-processing with {args.ParamFile}\n\n')
    main(params)
