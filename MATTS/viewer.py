import h5py
from matplotlib import pyplot as plt
import numpy as np
import cv2




print('\n\n\n')




# hdf5 OPTIONS :

data_location = '.\samples\\'

experiment_dir = 'samples256_overlap20_min-annot3_3bands_pls_work'
sets_to_look_at = ['trn', 'tst', 'val']
dataset_ims_to_show = ('map_img', 'sat_img')


for set in sets_to_look_at:
    f = h5py.File(data_location + experiment_dir + '\\' + set + '_samples.hdf5', 'r')

# PULL EACH SET --------------------------------------------------------------------------------------------------------
    dataset_names = list(f.keys())
    datasets = []
    for dataset_name in dataset_names:
        datasets.append(f[dataset_name])

# # PRINTS ---------------------------------------------------------------------------------------------------------------
#     print(set)
#     print('\tkeys =')
#     for d in range(len(dataset_names)):
#         print(' - ', dataset_names[d], ' - ', datasets[d].shape, ' - ', datasets[d].dtype)
# 
#     print(datasets[2])

# SHOW IMS -------------------------------------------------------------------------------------------------------------
    dpi = 1/plt.rcParams['figure.dpi']
    fig = plt.figure('dataset_ims' + str(dataset_ims_to_show), constrained_layout=True, figsize=(3000/dpi, 1000/dpi), dpi=dpi)
    gs = fig.add_gridspec(2, f['map_img'].shape[0])
    ax = {}

    for dataset_name in dataset_ims_to_show:
        dataset = f[dataset_name]
        print(dataset.shape[0])
        for imN in range(dataset.shape[0]):
            ax[dataset_name+str(imN)] = fig.add_subplot(gs[('map_img', 'sat_img').index(dataset_name), imN])
            ax[dataset_name + str(imN)].set_title(dataset_name + ' ' + str(imN), )
            ax[dataset_name + str(imN)].imshow(dataset[imN, ...])

    plt.show()

    print('\n\n\n')



########################################################################################################################



# # tiff OPTIONS :
#
# data_location = '.\data\samples\\'
#
# experiment_dir = 'samples256_overlap33_min-annot3_3bands_pls_work'
# im_names = ['np_input_image_0.tif', 'np_input_image_1.tif' , 'np_label_rasterized_0.tif' , 'np_label_rasterized_1.tif']
#
#
# yIndex, xIndex = 0,0
# dpi = 1/plt.rcParams['figure.dpi']
# fig = plt.figure('dataset_ims' + str(dataset_ims_to_show), constrained_layout=True, figsize=(1000/dpi, 1000/dpi), dpi=dpi)
# gs = fig.add_gridspec(2, len(im_names)//2)
# ax = {}
# for im_name in im_names:
#     im = cv2.cvtColor(cv2.imread(data_location + experiment_dir + '\\' + im_name), cv2.COLOR_BGR2RGB)
#
# # PRINTS ---------------------------------------------------------------------------------------------------------------
#     print(im_name[:-4])
#     print('\t', im.shape)
#     print('\t\t max & min = ', np.max(im), ' ', np.min(im))
#     print('\t\t\t', im.dtype)
#
#     print('\n\n\n')
# # SHOW IMS -------------------------------------------------------------------------------------------------------------
#
#     yIndex = 0 if im_names.index(im_name) < 2 else 1
#     xIndex = 0 if im_names.index(im_name) % 2 == 0 else 1
#
#     ax[str(yIndex) + str(xIndex)] = fig.add_subplot(gs[yIndex,xIndex])
#     ax[str(yIndex) + str(xIndex)].imshow(im)
#
# plt.show()