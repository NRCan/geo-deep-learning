import math
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt, gridspec, cm, colors
import csv

from utils.utils import minmax_scale, get_key_def


def grid_vis(input, label, output, heatmaps, classes):
    """Visualizes image samples"""

    num_tiles = (len(heatmaps)+3) # length of heatmps + input, label and output
    height = math.ceil(num_tiles/4)
    width = num_tiles if num_tiles < 4 else 4
    plt.figure(figsize=(width*6, height*6))
    grid_spec = gridspec.GridSpec(height, width)

    list_imgs_pil = [input, label, output]
    list_titles = ['input', 'label', 'output']

    for index, key in enumerate(heatmaps.keys()): # TODO: test if heatmaps is empty dict
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

def vis_from_batch(params, inputs, labels, outputs, batch_index, vis_path, dataset='', ep_num=None): #FIXME: document
    for samp_index, zipped in enumerate(zip(inputs, labels, outputs)):

        samp_index = samp_index + len(inputs) * batch_index
        input, label, output = zipped
        vis(params, input, label, output,
            vis_path=vis_path,
            sample_num=samp_index+1,
            dataset=dataset,
            ep_num=ep_num)

        max_num_samples = get_key_def('max_num_vis_samples', params['visualization'], 4)
        if (samp_index + 1) >= max_num_samples:
            break


def vis(params, input, label, output, vis_path, sample_num, dataset='', ep_num=''):
    '''
    :param input: (tensor) input as pytorch tensor, e.g. as returned by dataloader
    :param label: (tensor) label as pytorch tensor, e.g. as returned by dataloader
    :param output: (tensor) output as pytorch tensor, e.g. as returned by dataloader
    :param scale: (param) scale range used in sample preparation
    :param vis_path: path where visualisation images will be saved
    :param sample_num: index of sample if function is from for loop iterating through a batch or list of images.
    :param colormap_file: csv file containing 5 columns (class name, input grayscale value, out red value, out green value,
                     out blue value) and num_class rows
    :param heatmaps:
    :param name_suffix:
    :param grid:
    :return:
    '''
    scale = get_key_def('scale_data', params['global'], None)
    colormap_file = get_key_def('colormap_file', params['visualization'], None)
    heatmaps = get_key_def('heatmaps', params['visualization'], False)
    grid = get_key_def('grid', params['visualization'], False)

    assert vis_path.parent.is_dir()
    vis_path.mkdir(exist_ok=True)
    softmax = torch.nn.Softmax(dim=0)
    output = softmax(output)

    input = input.cpu().permute(1, 2, 0).numpy()  # channels last
    label = label.cpu().numpy()
    output = output.detach().cpu().permute(1, 2, 0).numpy()  # channels last
    if output.shape[2] != 5:
        warnings.warn('Visualization was hardcoded for 4-class tasks. Problems may occur.')
    output_argmax = np.argmax(output, axis=2)

    input = minmax_scale(img=input, orig_range=(scale[0], scale[1]), scale_range=(0, 255)) if scale else input
    if input.shape != 3:
        input = input[:, :, :3]  # take three first bands assuming they are RGB in correct order
    input_PIL = Image.fromarray(input.astype(np.uint8), mode='RGB')

    ignore_index = get_key_def('ignore_index', params['training'], -1)
    if ignore_index < 0:
        warnings.warn('Choose 255 as ignore_index to visualize. Problems may occur otherwise...')
    classes = ['background', 'vegetation', 'hydro', 'roads', 'buildings']
    colormap = ['#000000ff', '#00680dff', '#b2e0e6ff', '#990000ff', '#efcd08ff', 'white'] # White for ignore_index
    if colormap_file: #FIXME: work in progress
        with open(colormap_file, 'rt') as file:
            reader = csv.reader(file)
            classes = ['background']
            colormap = ['#000000ff']
            for class_name, input_val, output_val in reader:
                classes.append(class_name)
                colormap.append(output_val) #FIXME: what if input values are not ascending or contiguous?

    label_color = gray_to_color(label, colormap) # convert label and output to color with colormap #FIXME add background in colormap
    output_argmax_color = gray_to_color(output_argmax, colormap)
    label_PIL = Image.fromarray((label_color[:, :, :3] * 255).astype(np.uint8), mode='RGB')
    output_argmax_PIL = Image.fromarray((output_argmax_color[:, :, :3] * 255).astype(np.uint8), mode='RGB')

    if heatmaps: # save per class heatmap # FIXME: document this in README
        heatmaps = {}
        for i in range(output.shape[2]):  # for each channel (i.e. class) in output
            perclass_output = output[:, :, i]
            # perclass_output = minmax_scale(img=perclass_output, orig_range=(0, 1), scale_range=(0, 255))
            # https://stackoverflow.com/questions/10965417/how-to-convert-numpy-array-to-pil-image-applying-matplotlib-colormap
            perclass_output_PIL = Image.fromarray(np.uint8(cm.get_cmap('inferno')(perclass_output) * 255))
            heatmaps[i] = perclass_output_PIL
    else:
        heatmaps = None

    if grid:
        grid = grid_vis(input_PIL, label_PIL, output_argmax_PIL, heatmaps, classes)
        grid.savefig(vis_path.joinpath(f'{dataset}_{sample_num:03d}_ep{ep_num:03d}.png')) #FIXME change sample_index to sample_num
        plt.close()
    else:
        if not vis_path.joinpath(f'{dataset}_{sample_num:03d}_satimg.jpg').is_file():
            input_PIL.save(vis_path.joinpath(f'{dataset}_{sample_num:03d}_satimg.jpg'))
            label_PIL.save(vis_path.joinpath(f'{dataset}_{sample_num:03d}_label.png')) # save label
        output_argmax_PIL.save(vis_path.joinpath(f'{dataset}_{sample_num:03d}_output_ep{ep_num:03d}.png'))


def gray_to_color(graysc_array, custom_colormap=None):
    # classes = ['background', 'vegetation', 'hydro', 'roads', 'buildings']
    cmap = colors.ListedColormap(custom_colormap)
    color_array = cmap(graysc_array)
    background = np.asarray([[255, 255, 255]]) # background, black # FIXME integrate background
    return color_array