import shutil
from typing import Sequence, Union

import geopandas as gpd
import pyproj
import rasterio
import numpy as np
from shapely.geometry import box
from solaris.utils.core import _check_rasterio_im_load
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from omegaconf import DictConfig, open_dict, listconfig

# Our modules
from utils.logger import get_logger
from utils.geoutils import vector_to_raster
from utils.readers import image_reader_as_array
from utils.create_dataset import create_files_and_datasets, append_to_dataset
from utils.utils import (
    get_key_def, pad, pad_diff, read_csv, add_metadata_from_raster_to_sample, get_git_hash,
    read_modalities,
)
from utils.verifications import (
    validate_num_classes, assert_crs_match, validate_features_from_gpkg, validate_input_imagery, validate_by_rasterio,
    validate_by_geopandas
)
# Set the logging file
logging = get_logger(__name__)  # import logging

# Set random seed for reproducibility
np.random.seed(1234)


class AOI(object):
    """
    Object containing all data aoirmation about a single area of interest
    based on https://github.com/stac-extensions/ml-aoi
    """

    def __init__(self, raster: Union[Path, str],
                 label: Union[Path, str] = None,
                 split: str = None,
                 aoi_id: str = None,
                 collection: str = None,
                 attr_field_filter: str = None,
                 attr_values_filter: Sequence = None):
        """
        @param raster: pathlib.Path or str
            Path to source imagery
        @param label: pathlib.Path or str
            Path to ground truth file. If not provided, AOI is considered only for inference purposes
        @param split: str
            Name of destination dataset for aoi. Should be 'trn', 'tst' or 'inference'
        @param aoi_id: str
            Name or id (loosely defined) of area of interest. Used to name output folders.
            Multiple AOI instances can bear the same name.
        @param collection: str
            Name of collection containing AOI. All AOIs in the same collection should never be spatially overlapping
        @param attr_field_filter: str, optional
            Name of attribute field used to filter features. If not provided all geometries in ground truth file
            will be considered.
        @param attr_values_filter: list of ints, optional
            The list of attribute values in given attribute field used to filter features from ground truth file.
            If not provided, all values in field will be considered
        """
        validate_by_rasterio(raster)
        self.raster = _check_rasterio_im_load(raster)

        if label:
            validate_by_geopandas(label)
            label_bounds = gpd.read_file(label).total_bounds
            label_bounds_box = box(*label_bounds.tolist())
            raster_bounds_box = box(*list(self.raster.bounds))
            if not label_bounds_box.intersects(raster_bounds_box):
                raise ValueError(f"Features in label file {label} do not intersect with bounds of raster file "
                                 f"{raster.name}")
            self.label = Path(label)
            # may create overhead
            # TODO: unit test for failed CRS match
            try:
                self.crs_match, self.epsg_raster, self.epsg_label = assert_crs_match(self.raster, self.label)
            except pyproj.exceptions.CRSError as e:
                logging.warning(f"\nError while checking CRS match between raster and label."
                                f"\n{e}")
        else:
            self.label = label
            self.crs_match = self.epsg_raster = self.epsg_label = None

        if not isinstance(split, str) and split not in ['trn', 'tst', 'inference']:
            raise ValueError(f"\nDataset should be a string: 'trn', 'tst' or 'inference'. Got {split}.")
        elif not label and (split != 'inference' or not split):
            raise ValueError(f"\nNo ground truth provided. Dataset should be left empty or set to 'inference' only. "
                             f"\nGot {split}")
        self.split = split

        if aoi_id and not isinstance(aoi_id, str):
            raise TypeError(f'AOI name should be a string. Got {aoi_id} of type {type(aoi_id)}')
        elif not aoi_id:
            aoi_id = self.raster.stem  # Defaults to name of image without suffix
        self.aoi_id = aoi_id

        if collection and not isinstance(collection, str):
            raise TypeError(f'Collection name should be a string. Got {collection} of type {type(collection)}')
        self.aoi_id = aoi_id

        if label and attr_field_filter and not isinstance(attr_field_filter, str):
            raise TypeError(f'Attribute field name should be a string.\n'
                            f'Got {attr_field_filter} of type {type(attr_field_filter)}')
        self.attr_field_filter = attr_field_filter

        if label and attr_values_filter and not isinstance(attr_values_filter, (list, listconfig.ListConfig)):
            raise TypeError(f'Attribute values should be a list.\n'
                            f'Got {attr_values_filter} of type {type(attr_values_filter)}')
        self.attr_values_filter = attr_values_filter

    @classmethod
    def from_dict(cls, aoi_dict, attr_field_filter: str = None, attr_values_filter: list = None):
        """Instanciates an AOI object from an input-data dictionary as expected by geo-deep-learning"""
        if not isinstance(aoi_dict, dict):
            raise TypeError('Input data should be a dictionary.')
        # TODO: change dataset for split
        if not {'tif', 'gpkg', 'dataset'}.issubset(set(aoi_dict.keys())):
            raise ValueError(f"Input data should minimally contain the following keys: \n"
                             f"'tif', 'gpkg', 'dataset'.")
        if not aoi_dict['gpkg']:
            logging.warning(f"No ground truth data found for {aoi_dict['tif']}.\n"
                            f"Only imagery will be processed from now on")
        if "aoi_id" not in aoi_dict.keys() or not aoi_dict['aoi_id']:
            aoi_dict['aoi_id'] = Path(aoi_dict['tif']).stem
        aoi_dict['attribute_name'] = attr_field_filter
        new_aoi = cls(
            raster=aoi_dict['tif'],
            label=aoi_dict['gpkg'],
            split=aoi_dict['dataset'],
            attr_field_filter=attr_field_filter,
            attr_values_filter=attr_values_filter,
            aoi_id=aoi_dict['aoi_id']
        )
        return new_aoi

    def __str__(self):
        return (
            f"\nAOI ID: {self.aoi_id}"
            f"\n\tRaster: {self.raster.name}"
            f"\n\tLabel: {self.label}"
            f"\n\tCRS match: {self.crs_match}"
            f"\n\tSplit: {self.split}"
            f"\n\tAttribute field filter: {self.attr_field_filter}"
            f"\n\tAttribute values filter: {self.attr_values_filter}"
            )


def aois_from_csv(csv_path: Union[str, Path], attr_field_filter: str = None, attr_values_filter: str = None):
    """
    Creates list of AOIs by parsing a csv file referencing input data
    @param csv_path:
        path to csv file containing list of input data. See README for details on expected structure of csv.
    @param attr_values_filter:
        Attribute filed to filter features from
    @param attr_field_filter:
        Attribute values (for given attribute field) for features to keep
    Returns: a list of AOIs objects
    """
    aois = []
    data_list = read_csv(csv_path)
    logging.info(f'\n\tSuccessfully read csv file: {Path(csv_path).name}\n'
                 f'\tNumber of rows: {len(data_list)}\n'
                 f'\tCopying first row:\n{data_list[0]}\n')
    for i, aoi_dict in tqdm(enumerate(data_list), desc="Creating AOI's"):
        try:
            new_aoi = AOI.from_dict(
                aoi_dict=aoi_dict,
                attr_field_filter=attr_field_filter,
                attr_values_filter=attr_values_filter
            )
            logging.debug(new_aoi)
            aois.append(new_aoi)
        except FileNotFoundError as e:
            logging.critical(f"{e}\nGround truth file may not exist or is empty.\n"
                             f"Failed to create AOI:\n{aoi_dict}\n"
                             f"Index: {i}")
    return aois


def mask_image(arrayA, arrayB):
    """Function to mask values of arrayB, based on 0 values from arrayA.

    >>> x1 = np.array([0, 2, 4, 6, 0, 3, 9, 8], dtype=np.uint8).reshape(2,2,2)
    >>> x2 = np.array([1.5, 1.2, 1.6, 1.2, 11., 1.1, 25.9, 0.1], dtype=np.float32).reshape(2,2,2)
    >>> mask_image(x1, x2)
    array([[[ 0. ,  0. ],
            [ 1.6,  1.2]],
    <BLANKLINE>
           [[ 0. ,  0. ],
            [25.9,  0.1]]], dtype=float32)
    """

    # Handle arrayA of shapes (h,w,c) and (h,w)
    if len(arrayA.shape) == 3:
        mask = arrayA[:, :, 0] != 0
    else:
        mask = arrayA != 0

    ma_array = np.zeros(arrayB.shape, dtype=arrayB.dtype)
    # Handle arrayB of shapes (h,w,c) and (h,w)
    if len(arrayB.shape) == 3:
        for i in range(0, arrayB.shape[2]):
            ma_array[:, :, i] = mask * arrayB[:, :, i]
    else:
        ma_array = arrayB * mask
    return ma_array


def validate_class_prop_dict(actual_classes_dict, config_dict):
    """
    Populate dictionary containing class values found in vector data with values (thresholds) from sample/class_prop
    parameter in config file

    actual_classes_dict: dict
        Dictionary where each key is a class found in vector data. Value is not relevant (should be 0)

    config_dict:
        Dictionary with class ids (keys and thresholds (values) from config file

    """
    # Validation of class proportion parameters (assert types).
    if not config_dict:
        return None
    elif not isinstance(config_dict, dict):
        logging.warning(f"\nClass_proportion parameter should be a dictionary. Got type {type(config_dict)}")
        return None

    for key, value in config_dict.items():
        if not isinstance(key, str):
            raise TypeError(f"Class should be a string. Got {key} of type {type(key)}")
        try:
            int(key)
        except ValueError:
            raise ValueError('Class should be castable to an integer')
        if not isinstance(value, int):
            raise ValueError(f"Class value should be an integer, got {value} of type {type(value)}")

    # Populate actual classes dictionary with values from config
    for key, value in config_dict.items():
        if int(key) in actual_classes_dict.keys():
            actual_classes_dict[int(key)] = value
        else:
            logging.warning(f"\nClass {key} not found in provided vector data.")

    return actual_classes_dict.copy()


def minimum_annotated_percent(target_background_percent, min_annotated_percent):
    if not min_annotated_percent:
        return True
    elif float(target_background_percent) <= 100 - min_annotated_percent:
        return True

    return False


def class_proportion(target, sample_size: int, class_min_prop: dict):
    if not class_min_prop:
        return True
    sample_total = sample_size ** 2
    for key, value in class_min_prop.items():
        if key not in np.unique(target):
            target_prop_classwise = 0
        else:
            target_prop_classwise = (round((np.bincount(target.clip(min=0).flatten())[key] / sample_total) * 100, 1))
        if target_prop_classwise < value:
            return False
    return True


def add_to_datasets(dataset,
                    samples_file,
                    val_percent,
                    val_sample_file,
                    data,
                    target,
                    sample_metadata,
                    metadata_idx,
                    dict_classes,
                    stratification_bias=0,
                    stratification_dict=None):
    """ Add sample to Hdf5 (trn, val or tst) and computes pixel classes(%). """
    to_val_set = False
    if dataset == 'trn':
        random_val = np.random.randint(1, 100)
        if random_val > val_percent + stratification_bias:
            if stratification_dict is not None:
                stratification_dict['latest_assignment'] = 'trn'
        else:
            to_val_set = True
            samples_file = val_sample_file
            if stratification_dict is not None:
                stratification_dict['latest_assignment'] = 'val'
    append_to_dataset(samples_file["sat_img"], data)
    append_to_dataset(samples_file["map_img"], target)
    append_to_dataset(samples_file["sample_metadata"], repr(sample_metadata))
    append_to_dataset(samples_file["meta_idx"], metadata_idx)

    # adds pixel count to pixel_classes dict for each class in the image
    class_vals, counts = np.unique(target, return_counts=True)
    for i in range(len(class_vals)):
        class_val = class_vals[i]
        count = counts[i]
        dict_classes[class_val] += count
        if class_val not in dict_classes.keys():
            logging.error(f'Sample contains value "{class_val}" not defined in the classes ({dict_classes.keys()}).')
    return to_val_set


def samples_preparation(in_img_array,
                        label_array,
                        sample_size,
                        overlap,
                        samples_count,
                        num_classes,
                        samples_file,
                        val_percent,
                        val_sample_file,
                        dataset,
                        pixel_classes,
                        dontcare,
                        image_metadata=None,
                        min_annot_perc=None,
                        class_prop=None,
                        stratd=None):
    """
    Extract and write samples from input image and reference image
    :param in_img_array: numpy array of the input image
    :param label_array: numpy array of the annotation image
    :param sample_size: (int) Size (in pixel) of the samples to create # TODO: could there be a different sample size for tst dataset? shows results closer to inference
    :param overlap: (int) Desired overlap between samples in %
    :param samples_count: (dict) Current number of samples created (will be appended and return)
    :param num_classes: (dict) Number of classes in reference data (will be appended and return)
    :param samples_file: (hdf5 dataset) hdfs file where samples will be written
    :param val_percent: (int) percentage of validation samples
    :param val_sample_file: (hdf5 dataset) hdfs file where samples will be written (val)
    :param dataset: (str) Type of dataset where the samples will be written. Can be 'trn' or 'val' or 'tst'
    :param pixel_classes: (dict) samples pixel statistics
    :param image_metadata: (dict) metadata associated to source raster
    :param dontcare: Value in gpkg features that will ignored during training
    :param min_annot_perc: optional, minimum annotated percent required for sample to be created
    :param class_prop: optional, minimal proportion of pixels for each class required for sample to be created
    :return: updated samples count and number of classes.
    """

    # read input and reference images as array
    h, w, num_bands = in_img_array.shape
    if dataset == 'trn':
        idx_samples = samples_count['trn']
        append_to_dataset(val_sample_file["metadata"], repr(image_metadata))
    elif dataset == 'tst':
        idx_samples = samples_count['tst']
    else:
        raise ValueError(f"Dataset value must be trn or tst. Provided value is {dataset}")

    idx_samples_v = samples_count['val']

    # Adds raster metadata to the dataset. All samples created by tiling below will point to that metadata by index
    metadata_idx = append_to_dataset(samples_file["metadata"], repr(image_metadata))

    if overlap > 25:
        logging.warning(
            "\nhigh overlap >25%, note that automatic train/val split creates very similar samples in both sets"
        )
    dist_samples = round(sample_size * (1 - (overlap / 100)))
    added_samples = 0
    excl_samples = 0

    # with tqdm(range(0, h, dist_samples), position=1, leave=True,
    #           desc=f'Writing samples. Dataset currently contains {idx_samples} '
    #                f'samples') as _tqdm:
    with tqdm(range(0, h, dist_samples), position=1, leave=True) as _tqdm:
        for row in _tqdm:
            for column in range(0, w, dist_samples):
                data = (in_img_array[row:row + sample_size, column:column + sample_size, :])
                target = np.squeeze(label_array[row:row + sample_size, column:column + sample_size, :], axis=2)
                data_row = data.shape[0]
                data_col = data.shape[1]
                if data_row < sample_size or data_col < sample_size:
                    padding = pad_diff(
                        data_row, data_col, sample_size, sample_size  # array, actual height, actual width, desired size
                    )
                    # don't fill with 0 if possible. Creates false min value when scaling.
                    data = pad(data, padding, fill=np.nan)

                target_row = target.shape[0]
                target_col = target.shape[1]
                if target_row < sample_size or target_col < sample_size:
                    padding = pad_diff(target_row, target_col, sample_size,
                                       sample_size)  # array, actual height, actual width, desired size
                    target = pad(target, padding, fill=dontcare)
                backgr_ct = np.sum(target == 0)
                backgr_ct += np.sum(target == dontcare)
                target_background_percent = round(backgr_ct / target.size * 100, 1)

                sample_metadata = {'sample_indices': (row, column)}

                # Stratification bias
                if (stratd is not None) and (dataset == 'trn'):
                    tile_size = target.size
                    u, count = np.unique(target, return_counts=True)
                    tile_counts = {x: y for x, y in zip(u, count)}
                    tile_props = {x: y / tile_size for x, y in zip(u, count)}
                    for key in tile_props.keys():
                        if key not in stratd['trn']['total_counts']:
                            stratd['trn']['total_counts'][key] = 0
                        if key not in stratd['val']['total_counts']:
                            stratd['val']['total_counts'][key] = 0
                    if stratd['trn']['total_pixels'] == 0:
                        stratd['trn']['total_props'] = {key: 0.0 for key in stratd['trn']['total_counts'].keys()}
                    else:
                        stratd['trn']['total_props'] = {key: val / stratd['trn']['total_pixels']
                                                        for key, val in stratd['trn']['total_counts'].items()}
                    if stratd['val']['total_pixels'] == 0:
                        stratd['val']['total_props'] = {key: 0.0 for key in stratd['val']['total_counts'].keys()}
                    else:
                        stratd['val']['total_props'] = {key: val / stratd['val']['total_pixels']
                                                        for key, val in stratd['val']['total_counts'].items()}
                    distances_trn = {key: np.abs(val - stratd['trn']['total_props'][key])
                                     for key, val in tile_props.items()}
                    distances_val = {key: np.abs(val - stratd['val']['total_props'][key])
                                     for key, val in tile_props.items()}
                    dist_trn = np.mean(np.array(list(distances_trn.values())) ** 2)
                    dist_val = np.mean(np.array(list(distances_val.values())) ** 2)
                    dist = dist_val - dist_trn
                    stratification_bias = stratd['strat_factor'] * np.sign(dist)
                else:
                    stratification_bias = 0.0

                val = False
                if minimum_annotated_percent(target_background_percent, min_annot_perc) and \
                        class_proportion(target, sample_size, class_prop):
                    val = add_to_datasets(dataset=dataset,
                                          samples_file=samples_file,
                                          val_percent=val_percent,
                                          val_sample_file=val_sample_file,
                                          data=data,
                                          target=target,
                                          sample_metadata=sample_metadata,
                                          metadata_idx=metadata_idx,
                                          dict_classes=pixel_classes,
                                          stratification_bias=stratification_bias,
                                          stratification_dict=stratd)
                    if val:
                        idx_samples_v += 1
                    else:
                        idx_samples += 1
                    added_samples += 1

                    # Stratification update
                    if (stratd is not None) and (dataset == 'trn'):
                        for key, val in tile_counts.items():
                            stratd[stratd['latest_assignment']]['total_counts'][key] += val
                        stratd[stratd['latest_assignment']]['total_pixels'] += tile_size

                else:
                    excl_samples += 1

                target_class_num = np.max(target)
                if num_classes < target_class_num:
                    num_classes = target_class_num

                final_dataset = 'val' if val else dataset
                logging.debug(f'Dset={final_dataset}, '
                              f'Added samps={added_samples}/{len(_tqdm) * len(range(0, w, dist_samples))}, '
                              f'Excld samps={excl_samples}/{len(_tqdm) * len(range(0, w, dist_samples))}, '
                              f'Target annot perc={100 - target_background_percent:.1f}')

    if added_samples == 0:
        logging.warning(f"No sample added for current raster. Problems may occur with use of metadata")
    if dataset == 'tst':
        samples_count['tst'] = idx_samples
    else:
        samples_count['trn'] = idx_samples
        samples_count['val'] = idx_samples_v
    # return the appended samples count and number of classes.
    return samples_count, num_classes


def main(cfg: DictConfig) -> None:
    """
    Function that create training, validation and testing datasets preparation.

    1. Read csv file and validate existence of all input files and GeoPackages.
    2. Do the following verifications:
        1. Assert number of bands found in raster is equal to desired number
           of bands.
        2. Check that `num_classes` is equal to number of classes detected in
           the specified attribute for each GeoPackage.
           Warning: this validation will not succeed if a Geopackage
                    contains only a subset of `num_classes` (e.g. 3 of 4).
        3. Assert Coordinate reference system between raster and gpkg match.
    3. Read csv file and for each line in the file, do the following:
        1. Read input image as array with utils.readers.image_reader_as_array().
            - If gpkg's extent is smaller than raster's extent,
              raster is clipped to gpkg's extent.
            - If gpkg's extent is bigger than raster's extent,
              gpkg is clipped to raster's extent.
        2. Convert GeoPackage vector information into the "label" raster with
           utils.utils.vector_to_raster(). The pixel value is determined by the
           attribute in the csv file.
        3. Create a new raster called "label" with the same properties as the
           input image.
        4. Read metadata and add to input as new bands (*more details to come*).
        5. Crop the arrays in smaller samples of the size `samples_size` of
           `your_conf.yaml`. Visual representation of this is provided at
            https://medium.com/the-downlinq/broad-area-satellite-imagery-semantic-segmentation-basiss-4a7ea2c8466f
        6. Write samples from input image and label into the "val", "trn" or
           "tst" hdf5 file, depending on the value contained in the csv file.
            Refer to samples_preparation().

    -------
    :param cfg: (dict) Parameters found in the yaml config file.
    """
    # PARAMETERS
    num_classes = len(cfg.dataset.classes_dict.keys())
    num_bands = len(cfg.dataset.modalities)
    modalities = read_modalities(cfg.dataset.modalities)  # TODO add the Victor module to manage the modalities
    debug = cfg.debug

    # RAW DATA PARAMETERS
    data_path = get_key_def('raw_data_dir', cfg['dataset'], to_path=True, validate_path_exists=True)
    csv_file = get_key_def('raw_data_csv', cfg['dataset'], to_path=True, validate_path_exists=True)
    out_path = get_key_def('sample_data_dir', cfg['dataset'], default=data_path, to_path=True, validate_path_exists=True)

    # SAMPLE PARAMETERS
    samples_size = get_key_def('input_dim', cfg['dataset'], default=256, expected_type=int)
    overlap = get_key_def('overlap', cfg['dataset'], default=0)
    min_annot_perc = get_key_def('min_annotated_percent', cfg['dataset'], default=0)
    val_percent = get_key_def('train_val_percent', cfg['dataset'], default=0.3)['val'] * 100
    samples_folder_name = f'samples{samples_size}_overlap{overlap}_min-annot{min_annot_perc}' \
                          f'_{num_bands}bands_{cfg.general.project_name}'
    samples_dir = out_path.joinpath(samples_folder_name)
    if samples_dir.is_dir():
        if debug:
            # Move existing data folder with a random suffix.
            last_mod_time_suffix = datetime.fromtimestamp(samples_dir.stat().st_mtime).strftime('%Y%m%d-%H%M%S')
            shutil.move(samples_dir, out_path.joinpath(f'{str(samples_dir)}_{last_mod_time_suffix}'))
        else:
            logging.critical(
                f'Data path exists: {samples_dir}. Remove it or use a different experiment_name.'
            )
            raise FileExistsError()
    Path.mkdir(samples_dir, exist_ok=False)  # TODO: what if we want to append samples to existing hdf5?

    # LOGGING PARAMETERS  TODO see logging yaml
    experiment_name = cfg.general.project_name
    # mlflow_uri = get_key_def('mlflow_uri', params['global'], default="./mlruns")

    # OTHER PARAMETERS
    # TODO class_prop get_key_def('class_proportion', params['sample']['sampling_method'], None, expected_type=dict)
    class_prop = None
    mask_reference = False  # TODO get_key_def('mask_reference', params['sample'], default=False, expected_type=bool)
    # set dontcare (aka ignore_index) value
    dontcare = cfg.dataset.ignore_index if cfg.dataset.ignore_index is not None else -1
    if dontcare == 0:
        raise ValueError("\nThe 'dontcare' value (or 'ignore_index') used in the loss function cannot be zero.")
    attribute_field = get_key_def('attribute_field', cfg['dataset'], None, expected_type=str)
    # Assert that all items in attribute_values are integers (ex.: single-class samples from multi-class label)
    attr_vals = get_key_def('attribute_values', cfg['dataset'], None, expected_type=Sequence)
    if attr_vals is list:
        for item in attr_vals:
            if not isinstance(item, int):
                raise logging.critical(ValueError(f'\nAttribute value "{item}" is {type(item)}, expected int.'))

    # OPTIONAL
    use_stratification = cfg.dataset.use_stratification if cfg.dataset.use_stratification is not None else False
    if use_stratification:
        stratd = {
            'trn': {'total_pixels': 0, 'total_counts': {}, 'total_props': {}},
            'val': {'total_pixels': 0, 'total_counts': {}, 'total_props': {}},
            'strat_factor': cfg['dataset']['use_stratification']
        }
    else:
        stratd = None

    # ADD GIT HASH FROM CURRENT COMMIT TO PARAMETERS (if available and parameters will be saved to hdf5s).
    with open_dict(cfg):
        cfg.general.git_hash = get_git_hash()

    list_data_prep = aois_from_csv(
        csv_path=csv_file,
        attr_field_filter=attribute_field,
        attr_values_filter=attr_vals
    )

    # IF DEBUG IS ACTIVATE
    if debug:
        logging.warning(
            f'\nDebug mode activated. Some debug features may mobilize extra disk space and cause delays in execution.'
        )

    # VALIDATION: (1) Assert num_classes parameters == num actual classes in gpkg and (2) check CRS match (tif and gpkg)
    valid_gpkg_set = set()
    for aoi in tqdm(list_data_prep, position=0):
        validate_input_imagery(aoi.raster.name, num_bands)
        if aoi.label not in valid_gpkg_set:
            gpkg_classes = validate_num_classes(
                aoi.label, num_classes, attribute_field, dontcare, attribute_values=attr_vals,
            )
            assert_crs_match(aoi.raster.name, aoi.label)
            valid_gpkg_set.add(aoi.label)

    if debug:
        # VALIDATION (debug only): Checking validity of features in vector files
        for aoi in tqdm(list_data_prep, position=0, desc=f"Checking validity of features in vector files"):
            # TODO: make unit to test this with invalid features.
            invalid_features = validate_features_from_gpkg(aoi.label, attribute_field)
            if invalid_features:
                logging.critical(f"{aoi.label}: Invalid geometry object(s) '{invalid_features}'")

    number_samples = {'trn': 0, 'val': 0, 'tst': 0}
    number_classes = 0

    trn_hdf5, val_hdf5, tst_hdf5 = create_files_and_datasets(samples_size=samples_size,
                                                             number_of_bands=num_bands,
                                                             samples_folder=samples_dir,
                                                             cfg=cfg)

    # creates pixel_classes dict and keys
    pixel_classes = {key: 0 for key in gpkg_classes}
    background_val = 0
    pixel_classes[background_val] = 0
    class_prop = validate_class_prop_dict(pixel_classes, class_prop)
    pixel_classes[dontcare] = 0

    # For each row in csv: (1) burn vector file to raster, (2) read input raster image, (3) prepare samples
    logging.info(
        f"\nPreparing samples \n  Samples_size: {samples_size} \n  Overlap: {overlap} "
        f"\n  Validation set: {val_percent} % of created training samples"
    )
    for aoi in tqdm(list_data_prep, position=0, leave=False):
        try:
            logging.info(f"\nReading as array: {aoi.raster.name}")
            with _check_rasterio_im_load(aoi.raster) as raster:
                # 1. Read the input raster image
                np_input_image, raster, dataset_nodata = image_reader_as_array(
                    input_image=raster,
                    clip_gpkg=aoi.label
                )

                # 2. Burn vector file in a raster file
                logging.info(f"\nRasterizing vector file (attribute: {attribute_field}): {aoi.label}")
                try:
                    np_label_raster = vector_to_raster(vector_file=aoi.label,
                                                       input_image=raster,
                                                       out_shape=np_input_image.shape[:2],
                                                       attribute_name=attribute_field,
                                                       fill=background_val,
                                                       attribute_values=attr_vals)  # background value in rasterized vector.
                except ValueError:
                    logging.error(f"No vector features found for {aoi.label} with provided configuration."
                                  f"Will skip to next AOI.")
                    continue

                if dataset_nodata is not None:
                    # 3. Set ignore_index value in label array where nodata in raster (only if nodata across all bands)
                    np_label_raster[dataset_nodata] = dontcare

            if debug:
                out_meta = raster.meta.copy()
                np_image_debug = np_input_image.transpose(2, 0, 1).astype(out_meta['dtype'])
                out_meta.update({"driver": "GTiff",
                                 "height": np_image_debug.shape[1],
                                 "width": np_image_debug.shape[2]})
                out_tif = samples_dir / f"{Path(aoi.raster.name).stem}_clipped.tif"
                logging.debug(f"Writing clipped raster to {out_tif}")
                with rasterio.open(out_tif, "w", **out_meta) as dest:
                    dest.write(np_image_debug)

                out_meta = raster.meta.copy()
                np_label_debug = np.expand_dims(np_label_raster, axis=2).transpose(2, 0, 1).astype(out_meta['dtype'])
                out_meta.update({"driver": "GTiff",
                                 "height": np_label_debug.shape[1],
                                 "width": np_label_debug.shape[2],
                                 'count': 1})
                out_tif = samples_dir / f"{Path(aoi.label).stem}_clipped.tif"
                logging.debug(f"\nWriting final rasterized gpkg to {out_tif}")
                with rasterio.open(out_tif, "w", **out_meta) as dest:
                    dest.write(np_label_debug)

            # Mask the zeros from input image into label raster.
            if mask_reference:
                np_label_raster = mask_image(np_input_image, np_label_raster)

            if aoi.split == 'trn':
                out_file = trn_hdf5
            elif aoi.split == 'tst':
                out_file = tst_hdf5
            else:
                raise ValueError(f"\nDataset value must be trn or tst. Provided value is {aoi.split}")
            val_file = val_hdf5

            metadata = add_metadata_from_raster_to_sample(sat_img_arr=np_input_image,
                                                          raster_handle=raster)
            # Save label's per class pixel count to image metadata
            metadata['source_label_bincount'] = {class_num: count for class_num, count in
                                                 enumerate(np.bincount(np_label_raster.clip(min=0).flatten()))
                                                 if count > 0}  # TODO: add this to add_metadata_from[...] function?

            np_label_raster = np.reshape(np_label_raster, (np_label_raster.shape[0], np_label_raster.shape[1], 1))
            # 3. Prepare samples!
            number_samples, number_classes = samples_preparation(in_img_array=np_input_image,
                                                                 label_array=np_label_raster,
                                                                 sample_size=samples_size,
                                                                 overlap=overlap,
                                                                 samples_count=number_samples,
                                                                 num_classes=number_classes,
                                                                 samples_file=out_file,
                                                                 val_percent=val_percent,
                                                                 val_sample_file=val_file,
                                                                 dataset=aoi.split,
                                                                 pixel_classes=pixel_classes,
                                                                 dontcare=dontcare,
                                                                 image_metadata=metadata,
                                                                 min_annot_perc=min_annot_perc,
                                                                 class_prop=class_prop,
                                                                 stratd=stratd)

            # logging.info(f'\nNumber of samples={number_samples}')
            out_file.flush()
        except OSError:
            logging.exception(f'\nAn error occurred while preparing samples with "{Path(aoi["tif"]).stem}" (tiff) and '
                              f'{Path(aoi["gpkg"]).stem} (gpkg).')
            continue

    trn_hdf5.close()
    val_hdf5.close()
    tst_hdf5.close()

    pixel_total = 0
    # adds up the number of pixels for each class in pixel_classes dict
    for i in pixel_classes:
        pixel_total += pixel_classes[i]
    # calculate the proportion of pixels of each class for the samples created
    pixel_classes_dict = {}
    for i in pixel_classes:
        # prop = round((pixel_classes[i] / pixel_total) * 100, 1) if pixel_total > 0 else 0
        pixel_classes_dict[i] = round((pixel_classes[i] / pixel_total) * 100, 1) if pixel_total > 0 else 0
    # prints the proportion of pixels of each class for the samples created
    msg_pixel_classes = "\n".join("Pixels from class {}: {}%".format(k, v) for k, v in pixel_classes_dict.items())
    logging.info("\n" + msg_pixel_classes)

    logging.info(f"\nNumber of samples created: {number_samples}")

