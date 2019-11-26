import math
import warnings

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt, gridspec, cm

from utils.utils import minmax_scale


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
        list_imgs_pil.extend(heatmaps[key])
        list_titles.extend(classes[index])

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


def vis(input, label, output, scale, vis_path, index, colormap_file=None, heatmaps=True, name_suffix='', grid=True):
    '''
    :param input: (tensor) input as pytorch tensor, e.g. as returned by dataloader
    :param label: (tensor) label as pytorch tensor, e.g. as returned by dataloader
    :param output: (tensor) output as pytorch tensor, e.g. as returned by dataloader
    :param scale: (param) scale range used in sample preparation
    :param vis_path: path where visualisation images will be saved
    :param index:
    :param colormap_file: csv file containing 5 columns (class name, input grayscale value, out red value, out green value,
                     out blue value) and num_class rows
    :param heatmaps:
    :param name_suffix:
    :param grid:
    :return:
    '''
    softmax = torch.nn.Softmax(dim=0)
    output = softmax(output)

    input = input.cpu().permute(1, 2, 0).numpy()  # channels last
    label = label.cpu().numpy()
    output = output.cpu().permute(1, 2, 0).numpy()  # channels last
    output_argmax = np.argmax(output, axis=2)

    input = minmax_scale(img=input, orig_range=(scale[0], scale[1]), scale_range=(0, 255)) if scale else input
    if input.shape != 3:
        input = input[:, :, :3]  # take three first bands assuming they are RGB in correct order
    input_PIL = Image.fromarray(input.astype(np.uint8), mode='RGB')

    label = gray_to_color(label, colormap_file) # convert label and output to color with colormap #FIXME add background in colormap
    output_argmax = gray_to_color(output_argmax, colormap_file)
    label_PIL = Image.fromarray(label.astype(np.uint8), mode='RGB')
    output_argmax_PIL = Image.fromarray(output_argmax.astype(np.uint8), mode='RGB')

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
        if not colormap_file:
            classes = ['background', 'vegetation', 'hydro', 'roads', 'buildings']
        else:
            raise NotImplementedError('Importing class names from csv not implemented yet.')
        grid = grid_vis(input_PIL, label_PIL, output_argmax_PIL, heatmaps, classes)
        grid.savefig(vis_path.joinpath(f'{index:03d}_{name_suffix}.png'))
    else:
        input_PIL.save(vis_path.joinpath(f'{index:03d}_satimg.jpg'))
        label_PIL.save(vis_path.joinpath(f'{index:03d}_label.png')) # save label
        output_argmax_PIL.save(vis_path.joinpath(f'{index:03d}_output.png'))


def gray_to_color(graysc_array, colormap_file=None):
    if not colormap_file:
        colormap = np.asarray([
            [255, 255, 255],  # background, black
            [0, 104, 13],  # vegetation, green
            [178, 224, 230],  # hydro, blue
            [153, 0, 0],  # roads, red
            [239, 205, 8]])  # buildings, yellow
    else:
        warnings.warn('Loading external colormap is not yet implemented')
        colormap = np.genfromtxt(colormap_file, delimiter=';', skip_header=1, usecols=(1, 2, 3, 4))
        background = np.asarray([[255, 255, 255]]) # background, black # FIXME integrate background
    color_array = colormap[graysc_array]
    return color_array