import math
import re
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt, gridspec, cm, colors
import csv

from utils.utils import minmax_scale, get_key_def


def grid_vis(input, output, heatmaps, classes, label=None):
    """Visualizes image samples"""

    list_imgs_pil = [input, label, output] if label else [input, output]
    list_titles = ['input', 'label', 'output'] if label else ['input', 'output']

    num_tiles = (len(list_imgs_pil)+len(heatmaps))
    height = math.ceil(num_tiles/4)
    width = num_tiles if num_tiles < 4 else 4
    plt.figure(figsize=(width*6, height*6))
    grid_spec = gridspec.GridSpec(height, width)

    for index, key in enumerate(heatmaps.keys()):
        list_imgs_pil.append(heatmaps[key])
        list_titles.append(classes[index])

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

def vis_from_batch(params, inputs, outputs, batch_index, vis_path, labels=None, dataset='', ep_num=0): #FIXME: document
    """
    :param params:
    :param inputs:
    :param outputs:
    :param batch_index:
    :param vis_path:
    :param labels:
    :param dataset:
    :param ep_num:
    :return:
    """
    assert params['global']['task'] == 'segmentation'
    labels = [None]*(len(outputs)) if not labels else labels # Creaty empty list of labels to enable zip operation below if no label

    for samp_index, zipped in enumerate(zip(inputs, labels, outputs)):

        samp_index = samp_index + len(inputs) * batch_index
        input, label, output = zipped
        vis(params, input, output,
            vis_path=vis_path,
            sample_num=samp_index+1,
            label=label,
            dataset=dataset,
            ep_num=ep_num)

        max_num_samples = get_key_def('max_num_vis_samples', params['visualization'], 4)
        if (samp_index + 1) >= max_num_samples:
            break


def vis(params, input, output, vis_path, sample_num, label=None, dataset='', ep_num=0):
    '''
    :param params: parameters from .yaml config file
    :param input: (tensor) input array as pytorch tensor, e.g. as returned by dataloader
    :param output: (tensor) output array as pytorch tensor, e.g. as returned by dataloader
    :param vis_path: path where visualisation images will be saved
    :param sample_num: index of sample if function is from for loop iterating through a batch or list of images.
    :param label: (tensor) label array as pytorch tensor, e.g. as returned by dataloader. Optional.
    :param dataset: (str) name of dataset arrays belong to. For file-naming purposes only.
    :param ep_num: (int) number of epoch arrays are inputted from. For file-naming purposes only.
    :return: saves color images from input arrays as grid or as full scale .png
    '''
    scale = get_key_def('scale_data', params['global'], None)
    colormap_file = get_key_def('colormap_file', params['visualization'], None)
    heatmaps = get_key_def('heatmaps', params['visualization'], False)
    grid = get_key_def('grid', params['visualization'], False)
    ignore_index = get_key_def('ignore_index', params['training'], -1)

    assert vis_path.parent.is_dir()
    vis_path.mkdir(exist_ok=True)
    softmax = torch.nn.Softmax(dim=0)
    output = softmax(output)

    input = input.cpu().permute(1, 2, 0).numpy()  # channels last
    label = label.cpu().numpy() if label else None
    output = output.detach().cpu().permute(1, 2, 0).numpy()  # channels last

    # PREPARE HEATMAPS FROM SOFTMAX OUTPUT
    heatmaps_dict = {}
    if heatmaps:  # save per class heatmap  FIXME: document this in README
        for i in range(output.shape[2]):  # for each channel (i.e. class) in output
            perclass_output = output[:, :, i]
            # perclass_output = minmax_scale(img=perclass_output, orig_range=(0, 1), scale_range=(0, 255))
            # https://stackoverflow.com/questions/10965417/how-to-convert-numpy-array-to-pil-image-applying-matplotlib-colormap
            perclass_output_PIL = Image.fromarray(np.uint8(cm.get_cmap('inferno')(perclass_output) * 255))
            heatmaps_dict[i] = perclass_output_PIL

    output_argmax = np.argmax(output, axis=2)

    input = minmax_scale(img=input, orig_range=(scale[0], scale[1]), scale_range=(0, 255)) if scale else input
    if input.shape != 3:
        input = input[:, :, :3]  # take three first bands assuming they are RGB in correct order
    input_PIL = Image.fromarray(input.astype(np.uint8), mode='RGB')

    if label and ignore_index < 0:  # TODO: test when ignore_index is smaller than 1.
        warnings.warn('Choose 255 as ignore_index to visualize. Problems may occur otherwise...')
        label[label == ignore_index] = 255  # Convert all pixels with ignore_index values to 255 to make sure it is last in order of values.

    # DEFINE COLORMAP AND NAMES OF CLASSES WITH RESPECT TO GRAYSCALE VALUES
    if colormap_file:
        classes, html_colors = colormap_reader(colormap_file)
        assert len(html_colors) >= len(np.unique(output_argmax))
        if label and ignore_index in np.unique(label):
            html_colors.append('white')  # for ignore_index values in labels. #TODO: test this.
        cmap = colors.ListedColormap(html_colors)
    else:
        classes = range(0, output.shape[2])
        cmap = cm.get_cmap('Set1')

    # CONVERT OUTPUT AND LABEL, IF PROVIDED, TO RGB WITH MATPLOTLIB'S COLORMAP OBJECT
    output_argmax_color = cmap(output_argmax)
    output_argmax_PIL = Image.fromarray((output_argmax_color[:, :, :3] * 255).astype(np.uint8), mode='RGB')
    if label:
        label_color = cmap(label)
        label_PIL = Image.fromarray((label_color[:, :, :3] * 255).astype(np.uint8), mode='RGB')
    else:
        label_PIL = None

    if grid:  # SAVE PIL IMAGES AS GRID
        grid = grid_vis(input_PIL, output_argmax_PIL, heatmaps_dict, classes, label=label_PIL)
        grid.savefig(vis_path.joinpath(f'{dataset}_{sample_num:03d}_ep{ep_num:03d}.png'))
        plt.close()
    else:  # SAVE PIL IMAGES DIRECTLY TO FILE
        if not vis_path.joinpath(f'{dataset}_{sample_num:03d}_satimg.jpg').is_file():
            input_PIL.save(vis_path.joinpath(f'{dataset}_{sample_num:03d}_satimg.jpg'))
            if label:
                label_PIL.save(vis_path.joinpath(f'{dataset}_{sample_num:03d}_label.png')) # save label
        output_argmax_PIL.save(vis_path.joinpath(f'{dataset}_{sample_num:03d}_output_ep{ep_num:03d}.png'))


def colormap_reader(colormap_path):
    """
    :param colormap_path: csv file (with header) containing 3 columns (input grayscale value, classes, html colors (#RRGGBB))
    :return: list of classes and list of html colors to map to grayscale values associated with classes
    """
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

    return classes_list, html_colors