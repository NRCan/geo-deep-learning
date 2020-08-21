import math
import re
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from matplotlib import pyplot as plt, gridspec, cm, colors
import csv

from utils.utils import unscale, unnormalize, get_key_def
from utils.geoutils import create_new_raster_from_base


def grid_vis(input, output, heatmaps_dict, label=None, heatmaps=True):
    """ Create a grid with PIL images and titles
    :param input: (tensor) input array as pytorch tensor, e.g. as returned by dataloader
    :param output: (tensor) output array as pytorch tensor, e.g. as returned by dataloader
    :param heatmaps_dict: (dict) Dictionary of heatmaps where key is grayscale value of class and value a dict {'class_name': (str), 'heatmap_PIL': (PIL object))
    :param label: (tensor) label array as pytorch tensor, e.g. as returned by dataloader (optional)
    :param heatmaps: (bool) if True, include heatmaps in grid
    :return: Saves .png to disk
    """

    list_imgs_pil = [input, label, output] if label is not None else [input, output]
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


def vis_from_batch(params, inputs, outputs, batch_index, vis_path, labels=None, dataset='', ep_num=0, debug=False):
    """ Provide indiviual input, output and label from batch to visualization function
    :param params: (dict) Parameters found in the yaml config file.
    :param inputs: (tensor) inputs as pytorch tensors with dimensions (batch_size, channels, width, height)
    :param outputs: (tensor) outputs as pytorch tensors with dimensions (batch_size, channels, width, height)
    :param batch_index: (int) index of batch inside epoch
    :param vis_path: path where visualisation images will be saved
    :param labels: (tensor) labels as pytorch tensors with dimensions (batch_size, channels, width, height)
    :param dataset: name of dataset for file naming purposes (ex. 'tst')
    :param ep_num: (int) number of epoch for file naming purposes
    :param debug: (bool) if True, some debug features will be activated
    :return:
    """
    assert params['global']['task'] == 'segmentation'
    labels = [None]*(len(outputs)) if labels is None else labels  # Creaty empty list of labels to enable zip operation below if no label

    for batch_samp_index, zipped in enumerate(zip(inputs, labels, outputs)):
        epoch_samp_index = batch_samp_index + len(inputs) * batch_index
        input, label, output = zipped
        vis(params, input, output,
            vis_path=vis_path,
            sample_num=epoch_samp_index+1,
            label=label,
            dataset=dataset,
            ep_num=ep_num,
            debug=debug)


def vis(params, input, output, vis_path, sample_num=0, label=None, dataset='', ep_num=0, inference_input_path=False, debug=False):
    """saves input, output and label (if given) as .png in a grid or as individual pngs
    :param params: parameters from .yaml config file
    :param input: (tensor) input array as pytorch tensor, e.g. as returned by dataloader
    :param output: (tensor) output array as pytorch tensor before argmax, e.g. as returned by dataloader
    :param vis_path: path where visualisation images will be saved
    :param sample_num: index of sample if function is from for loop iterating through a batch or list of images.
    :param label: (tensor) label array as pytorch tensor, e.g. as returned by dataloader. Optional.
    :param dataset: (str) name of dataset arrays belong to. For file-naming purposes only.
    :param ep_num: (int) number of epoch arrays are inputted from. For file-naming purposes only.
    :param inference_input_path: (Path) path to input image on which inference is being performed. If given, turns «inference» bool to True below.
    :return: saves color images from input arrays as grid or as full scale .png
    """
    inference = True if inference_input_path else False
    scale = get_key_def('scale_data', params['global'], None)
    colormap_file = get_key_def('colormap_file', params['visualization'], None)
    heatmaps = get_key_def('heatmaps', params['visualization'], False)
    heatmaps_inf = get_key_def('heatmaps', params['inference'], False)
    grid = get_key_def('grid', params['visualization'], False)
    ignore_index = get_key_def('ignore_index', params['training'], -1)
    mean = get_key_def('mean', params['training']['normalization'])
    std = get_key_def('std', params['training']['normalization'])

    assert vis_path.parent.is_dir()
    vis_path.mkdir(exist_ok=True)

    if not inference:  # FIXME: function parameters should not come in as different types if inference or not.
        input = input.cpu().permute(1, 2, 0).numpy()  # channels last
        output = F.softmax(output, dim=0)  # Inference output is already softmax
        output = output.detach().cpu().permute(1, 2, 0).numpy()  # channels last
        if label is not None:
            label_copy = label.cpu().numpy().copy()
            if ignore_index < 0:
                warnings.warn('Choose 255 as ignore_index to visualize. Problems may occur otherwise...')
                new_ignore_index = 255
                # Convert all pixels with ignore_index values to 255 to make sure it is last in order of values.
                label_copy[label_copy == ignore_index] = new_ignore_index

    norm_mean = get_key_def('mean', params['training']['normalization'])
    norm_std = get_key_def('std', params['training']['normalization'])

    if norm_mean and norm_std:
        input = unnormalize(input_img=input, mean=mean, std=std)
    input = unscale(img=input, float_range=(scale[0], scale[1]), orig_range=(0, 255)) if scale else input
    if 1 <= input.shape[2] <= 2:
        input = input[:, :, :1]  # take first band (will become grayscale image)
        input = np.squeeze(input)
    elif input.shape[2] >= 3:
        input = input[:, :, :3]  # take three first bands assuming they are RGB in correct order
    mode = 'L' if input.shape[2] == 1 else 'RGB' # https://pillow.readthedocs.io/en/3.1.x/handbook/concepts.html#concept-modes
    input_PIL = Image.fromarray(input.astype(np.uint8), mode=mode)  # TODO: test this with grayscale input.

    # Give value of class to band with highest value in final inference
    output_argmax = np.argmax(output, axis=2).astype(np.uint8)  # Flatten along channels axis. Convert to 8bit

    # Define colormap and names of classes with respect to grayscale values
    classes, cmap = colormap_reader(output, colormap_file, default_colormap='Set1')

    heatmaps_dict = heatmaps_to_dict(output, classes, inference=inference, debug=debug)  # Prepare heatmaps from softmax output

    # Convert output and label, if provided, to RGB with matplotlib's colormap object
    output_argmax_color = cmap(output_argmax)
    output_argmax_PIL = Image.fromarray((output_argmax_color[:, :, :3] * 255).astype(np.uint8), mode='RGB')
    if not inference and label is not None:
        label_color = cmap(label_copy)
        label_PIL = Image.fromarray((label_color[:, :, :3] * 255).astype(np.uint8), mode='RGB')
    else:
        label_PIL = None

    if inference:
        if debug and len(np.unique(output_argmax)) == 1:
            warnings.warn(f'Inference contains only {np.unique(output_argmax)} value. Make sure data scale '
                          f'{scale} is identical with scale used for training model.')
        output_name = vis_path.joinpath(f"{inference_input_path.stem}_inference.tif")
        create_new_raster_from_base(inference_input_path, output_name, output_argmax)

        if heatmaps_inf:
            for key in heatmaps_dict.keys():
                heatmap = np.array(heatmaps_dict[key]['heatmap_PIL'])
                class_name = heatmaps_dict[key]['class_name']
                heatmap_name = vis_path.joinpath(f"{inference_input_path.stem}_inference_heatmap_{class_name}.tif")
                create_new_raster_from_base(inference_input_path, heatmap_name, heatmap)
    elif grid:  # SAVE PIL IMAGES AS GRID
        grid = grid_vis(input_PIL, output_argmax_PIL, heatmaps_dict, label=label_PIL, heatmaps=heatmaps)
        grid.savefig(vis_path.joinpath(f'{dataset}_{sample_num:03d}_ep{ep_num:03d}.png'))
        plt.close()
    else:  # SAVE PIL IMAGES DIRECTLY TO FILE
        if not vis_path.joinpath(f'{dataset}_{sample_num:03d}_satimg.jpg').is_file():
            input_PIL.save(vis_path.joinpath(f'{dataset}_{sample_num:03d}_satimg.jpg'))
            if not inference and label is not None:
                label_PIL.save(vis_path.joinpath(f'{dataset}_{sample_num:03d}_label.png'))  # save label
        output_argmax_PIL.save(vis_path.joinpath(f'{dataset}_{sample_num:03d}_output_ep{ep_num:03d}.png'))
        if heatmaps: # TODO: test this.
            for key in heatmaps_dict.keys():
                heatmap = heatmaps_dict[key]['heatmap_PIL']
                class_name = heatmaps_dict[key]['class_name']
                heatmap.save(vis_path.joinpath(f"{dataset}_{sample_num:03d}_output_ep{ep_num:03d}_heatmap_{class_name}.png"))  # save heatmap


def heatmaps_to_dict(output, classes=[], inference=False, debug=False):
    ''' Store heatmap into a dictionary
    :param output: softmax tensor
    :return: dictionary where key is value of class and value is numpy array
    '''
    heatmaps_dict = {}
    classes = range(output.shape[2]) if len(classes) == 0 else classes
    for i in range(output.shape[2]):  # for each channel (i.e. class) in output
        perclass_output = output[:, :, i]
        if inference:  # Don't color heatmap if in inference
            if debug:
                print(f'Heatmap class: {classes[i]}\n')
                print(f'List of unique values in heatmap: {np.unique(np.uint8(perclass_output * 255))}\n')
            perclass_output_PIL = Image.fromarray(np.uint8(perclass_output*255))
        else:  # https://stackoverflow.com/questions/10965417/how-to-convert-numpy-array-to-pil-image-applying-matplotlib-colormap
            perclass_output_PIL = Image.fromarray(np.uint8(cm.get_cmap('inferno')(perclass_output) * 255))
        heatmaps_dict[i] = {'class_name': classes[i], 'heatmap_PIL': perclass_output_PIL}

    return heatmaps_dict


def colormap_reader(output, colormap_path=None, default_colormap='Set1'):
    """
    :param colormap_path: csv file (with header) containing 3 columns (input grayscale value, classes, html colors (#RRGGBB))
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
        assert len(html_colors) == len(classes_list) >= output.shape[2], f'Not enough colors and class names for number of classes in output'
        html_colors.append('white')  # for ignore_index values in labels. #TODO: test this with a label containt ignore_index values
        cmap = colors.ListedColormap(html_colors)
    else:
        classes_list = list(range(0, output.shape[2]))  # TODO: since list of classes are only useful for naming each heatmap, this list could be inside the heatmaps_dict, e.g. {1: {heatmap: perclass_output_PIL, class_name: 'roads'}, ...}
        cmap = cm.get_cmap(default_colormap)

    return classes_list, cmap