import logging
import math
import re
from pathlib import Path
from typing import List, Dict, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from matplotlib import pyplot as plt, gridspec, colors
from matplotlib.colors import ListedColormap
import csv

from utils.utils import unscale, unnormalize, minmax_scale
from utils.geoutils import create_new_raster_from_base

import matplotlib

matplotlib.use('Agg')

logging.getLogger(__name__)


def grid_vis(input_, output, heatmaps_dict, label=None, heatmaps=True):
    """ Create a grid with PIL images and titles
    :param input_: (tensor) input array as pytorch tensor, e.g. as returned by dataloader
    :param output: (tensor) output array as pytorch tensor, e.g. as returned by dataloader
    :param heatmaps_dict: (dict) Dictionary of heatmaps where key is grayscale value of class and value a dict {'class_name': (str), 'heatmap_PIL': (PIL object))
    :param label: (tensor) label array as pytorch tensor, e.g. as returned by dataloader (optional)
    :param heatmaps: (bool) if True, include heatmaps in grid
    :return: Saves .png to disk
    """

    list_imgs_pil = [input_, label, output] if label is not None else [input_, output]
    list_titles = ['input', 'label', 'output'] if label is not None else ['input', 'output']

    num_tiles = (len(list_imgs_pil) + len(heatmaps_dict))
    height = math.ceil(num_tiles/4)
    width = num_tiles if num_tiles < 4 else 4
    plt.figure(figsize=(width*6, height*6))
    grid_spec = gridspec.GridSpec(height, width)

    if heatmaps:
        for key in heatmaps_dict.keys():
            list_imgs_pil.append(heatmaps_dict[key]['heatmap_PIL'])
            list_titles.append(heatmaps_dict[key]['class_name'])

    assert len(list_imgs_pil) == len(list_titles)
    for index, zipped in enumerate(zip(list_imgs_pil, list_titles)):
        img, title = zipped
        plt.subplot(grid_spec[index])
        plt.imshow(img)
        plt.grid(False)
        plt.axis('off')
        plt.title(title)
        plt.tight_layout()

    return plt


def vis_from_batch(vis_params: Dict,
                   inputs: torch.Tensor,
                   outputs: torch.Tensor,
                   batch_index: int,
                   vis_path: Union[str, Path],
                   labels: torch.Tensor = None,
                   dataset: str = '',
                   ep_num: int = 0,
                   scale: List = None,
                   debug: bool = False):
    """ Provide individual input, output and label from batch to visualization function
    :param vis_params: (Dict) parameters useful during visualization
    :param inputs: (tensor) inputs as pytorch tensors with dimensions (batch_size, channels, width, height)
    :param outputs: (tensor) outputs as pytorch tensors with dimensions (batch_size, channels, width, height)
    :param batch_index: (int) index of batch inside epoch
    :param vis_path: path where visualization images will be saved
    :param labels: (tensor) labels as pytorch tensors with dimensions (batch_size, channels, width, height)
    :param dataset: name of dataset for file naming purposes (ex. 'tst')
    :param ep_num: (int) number of epoch for file naming purposes
    :param scale: scale range
    :param debug: (bool) if True, some debug features will be activated
    :return:
    """
    # Create an empty list of labels to enable zip operation below if no label:
    labels = [None]*(outputs[0]) if labels is None else labels

    for batch_sample_index, data in enumerate(zip(inputs, labels, outputs)):
        epoch_sample_index = batch_sample_index + len(inputs) * batch_index
        image, label, output = data
        vis(vis_params=vis_params,
            image=image,
            label=label,
            output=output,
            vis_path=vis_path,
            sample_num=epoch_sample_index+1,
            dataset=dataset,
            ep_num=ep_num,
            scale=scale,
            debug=debug
            )


def vis(vis_params: Dict,
        image: torch.Tensor,
        output: torch.Tensor,
        label: torch.Tensor,
        vis_path: Union[str, Path],
        sample_num: int = 0,
        dataset: str = '',
        ep_num: int = 0,
        inference_input_path: Union[str, Path] = None,
        scale: List = None,
        debug: bool = False) -> None:
    """
    Saves input, output and label (if given) as .png in a grid or as individual pngs
    :param vis_params: (dict) visualization parameters
    :param image: (tensor) W/122input array as pytorch tensor, e.g. as returned by dataloader
    :param output: (tensor) output array as pytorch tensor before argmax, e.g. as returned by dataloader
    :param vis_path: path where visualization images will be saved
    :param sample_num: index of sample if function is from for loop iterating through a batch or list of images.
    :param label: (tensor) label array as pytorch tensor, e.g. as returned by dataloader. Optional.
    :param dataset: (str) name of dataset arrays belong to. For file-naming purposes only.
    :param ep_num: (int) number of epoch arrays are inputted from. For file-naming purposes only.
    :param inference_input_path: (Path) path to input image on which inference is being performed.
    If given, turns «inference» bool to True below.
    :param scale: scale range
    :param debug: True or False, for debugging
    :return: saves color images from input arrays as grid or as full scale .png
    """
    # Copy to cpu() memory and permute the dimensions: channel(s) last
    image = image.cpu().permute(1, 2, 0).numpy()
    n_classes = output.shape[0]

    assert vis_path.parent.is_dir(), "The parent folder for the visualization outputs does not exist."
    vis_path.mkdir(exist_ok=True)

    if n_classes == 1:
        output = torch.sigmoid(output)
        output_labels = torch.round(output)
        output_labels = np.array(output_labels.cpu()).squeeze(axis=0).astype(np.uint8)

    else:
        output = F.softmax(output, dim=0)
        output_labels = np.array(np.argmax(output.cpu(), axis=0)).astype(np.uint8)
    output = output.detach().cpu().permute(1, 2, 0).numpy()

    if label is not None:
        label_copy = label.cpu().numpy().copy()
        if vis_params['ignore_index'] < 0:
            new_ignore_index = 255
            # Convert all pixels with ignore_index values to 255 to make sure it is last in order of values.
            label_copy[label_copy == vis_params['ignore_index']] = new_ignore_index

    # Unnormalize and unscale the input image:
    if vis_params['mean'] and vis_params['std']:
        image = unnormalize(input_img=image, mean=vis_params['mean'], std=vis_params['std'])
    image = unscale(img=image, float_range=(scale[0], scale[1]), orig_range=(0, 255)) if scale else image

    # Create a PIL object for the input image:
    if 1 <= image.shape[2] <= 2:
        image = np.squeeze(image[:, :, :1], axis=2)  # take first band (will become grayscale image)
    elif image.shape[2] >= 3:
        image = image[:, :, :3]  # take three first bands assuming they are RGB in correct order
    mode = "L" if len(image.shape) == 2 else "RGB"
    image_pil = Image.fromarray(image.astype(np.uint8), mode=mode)

    # Define colormap and names of classes with respect to grayscale values
    classes, cmap = colormap_reader(
        n_classes=n_classes,
        colormap_path=vis_params['colormap_file'],
        default_colormap='Set1'
    )

    # Prepare heatmaps from softmax output
    heatmaps_dict = heatmaps_to_dict(
        output=output,
        classes=classes,
        inference=inference_input_path,
        debug=debug
    )

    # Convert output and label, if provided, to RGB with matplotlib's colormap object
    output_color = cmap(output_labels)
    output_pil = Image.fromarray((output_color[:, :, :3] * 255).astype(np.uint8), mode='RGB')

    label_color = cmap(label_copy)
    label_pil = Image.fromarray((label_color[:, :, :3] * 255).astype(np.uint8), mode='RGB')

    if inference_input_path is not None:
        if debug and len(np.unique(output)) == 1:
            logging.warning(f'Inference contains only {np.unique(output)} value. Make sure data scale '
                          f'{scale} is identical with scale used for training model.')
        output_name = vis_path.joinpath(f"{inference_input_path.stem}_inference.tif")
        create_new_raster_from_base(inference_input_path, output_name, output)

        if vis_params['heatmaps_inf']:
            for key in heatmaps_dict.keys():
                heatmap = np.array(heatmaps_dict[key]['heatmap_PIL'])
                class_name = heatmaps_dict[key]['class_name']
                heatmap_name = vis_path.joinpath(f"{inference_input_path.stem}_inference_heatmap_{class_name}.tif")
                create_new_raster_from_base(inference_input_path, heatmap_name, heatmap)

    elif vis_params['grid']:  # SAVE PIL IMAGES AS GRID
        grid = grid_vis(image_pil, output_pil, heatmaps_dict, label=label_pil, heatmaps=vis_params['heatmaps'])
        grid.savefig(vis_path.joinpath(f'{dataset}_{sample_num:03d}_ep{ep_num:03d}.png'))
        plt.close()

    else:  # SAVE PIL IMAGES DIRECTLY TO FILE
        if not vis_path.joinpath(f'{dataset}_{sample_num:03d}_satimg.jpg').is_file():
            image_pil.save(vis_path.joinpath(f'{dataset}_{sample_num:03d}_satimg.jpg'))
            if not inference_input_path and label is not None:
                label_pil.save(vis_path.joinpath(f'{dataset}_{sample_num:03d}_label.png'))  # save label
        output_pil.save(vis_path.joinpath(f'{dataset}_{sample_num:03d}_output_ep{ep_num:03d}.png'))

        if vis_params['heatmaps']:  # TODO: test this.
            for key in heatmaps_dict.keys():
                heatmap = heatmaps_dict[key]['heatmap_PIL']
                class_name = heatmaps_dict[key]['class_name']
                heatmap.save(vis_path.joinpath(f"{dataset}_{sample_num:03d}_output_ep{ep_num:03d}_heatmap_{class_name}.png"))  # save heatmap


def heatmaps_to_dict(output: np.ndarray,
                     classes: list,
                     inference: str = None,
                     debug: bool = False
                     ) -> Dict:
    """
    Store heatmap into a dictionary
    :param output: softmax numpy array
    :param classes: list of segmentation classes
    :param inference: target path for inferences
    :param debug: False or True for debugging
    :return: dictionary where key is value of class and value is numpy array
    """
    heatmaps_dict = {}
    classes = range(output.shape[2]) if len(classes) == 0 else classes
    for i in range(output.shape[2]):  # for each channel (i.e. class) in output
        perclass_output = output[:, :, i]
        if inference is not None:  # Don't color heatmap if in inference
            if debug:
                logging.info(f'Heatmap class: {classes[i]}\n')
                logging.info(f'List of unique values in heatmap: {np.unique(np.uint8(perclass_output * 255))}\n')
            perclass_output_pil = Image.fromarray(np.uint8(perclass_output*255))
        else:
            perclass_output_pil = Image.fromarray(np.uint8(plt.get_cmap('inferno')(perclass_output) * 255))
        heatmaps_dict[i] = {'class_name': classes[i], 'heatmap_PIL': perclass_output_pil}

    return heatmaps_dict


def colormap_reader(n_classes: int,
                    colormap_path: str = None,
                    default_colormap: str = 'Set1'
                    ) -> tuple[List, ListedColormap]:
    """
    :param n_classes: number of target classes
    :param colormap_path: csv file (with header) containing 3 columns (input grayscale value, classes, html colors (#RRGGBB))
    :param default_colormap: default color scheme from the available in matplotlib
    :return: list of classes and list of html colors to map to grayscale values associated with classes
    """
    if colormap_path is not None:
        assert Path(colormap_path).is_file(), f'Could not locate {colormap_path}'
        input_val = []
        classes_list = ['background']
        html_colors = ['#000000']
        with open(colormap_path, 'rt') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            rows = list(reader)
        input_val.extend([int(row[0]) for row in rows])
        csv_classes = [row[1] for row in rows]  # Take second element in row. Should be class name
        csv_html_colors = [row[2] for row in rows]  # Take third element in row. Should be hex color code
        sorted_classes = [x for _, x in sorted(zip(input_val, csv_classes))]  # sort according to grayscale values order
        sorted_colors = [x for _, x in sorted(zip(input_val, csv_html_colors))]
        for color in sorted_colors:
            match = re.search(r'^#(?:[0-9a-fA-F]{3}){1,2}$', color)
            assert match, f'Submitted color {color} does not match HEX color code pattern'
        classes_list.extend(sorted_classes)
        html_colors.extend(sorted_colors)
        assert len(html_colors) == len(classes_list) >= n_classes, f'Not enough colors and class names for number of classes in output'
        html_colors.append('white')  # for ignore_index values in labels. #TODO: test this with a label containt ignore_index values
        cmap = colors.ListedColormap(html_colors)
    else:
        classes_list = list(range(0, n_classes))  # TODO: since list of classes are only useful for naming each heatmap, this list could be inside the heatmaps_dict, e.g. {1: {heatmap: perclass_output_PIL, class_name: 'roads'}, ...}
        cmap = plt.get_cmap(default_colormap)

    return classes_list, cmap