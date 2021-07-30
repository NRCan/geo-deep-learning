import geopandas
import argparse, h5py
from pathlib import Path
from ruamel_yaml import YAML
from models.model_choice import net, load_checkpoint
from utils.utils import load_from_checkpoint
from matplotlib import pyplot as plt

# import numpy as np
# import cv2
from rich import print, inspect


'''



                                          ##### EXPORT TO GPKG INFO #####
'''
# TODO: add gpkg_index hdf5 datase
#       correlate each to txt file
#                                   line(index) = gpkg_name,projection,sample_index ( gotta split up trn/txt/val somehow)
'''
:::OPTIONS:::
        feature.geometry.type(s):(aka. shapely obj)
                                  -point
                                  -linestring
                                  -polygon
                                  -multipoint
                                  -multilinestring
                                  -multipolygon
                                  -geometrycollection
:::EXAMPLE:::
        # feature_coll = {
        #     "type": "FeatureCollection",
        #     "features": [
        #                 {
        #                     "id": "0",
        #                     "type": "Feature",
        #                     "properties": {"col1": "name1"},
        #                     "geometry": {"type": "Point",
        #                                  "coordinates": (1.0, 2.0)},
        #                     "bbox": (1.0, 2.0, 1.0, 2.0),
        #                 },
        #                 {
        #                     "id": "1",
        #                     "type": "Feature",
        #                     "properties": {"col1": "name2"},
        #                     "geometry": {"type": "Point",
        #                                  "coordinates": (2.0, 1.0)},
        #                     "bbox": (2.0, 1.0, 2.0, 1.0),
        #                 },
        #                 ],
        #     "bbox": (1.0, 1.0, 2.0, 2.0),  *** what is vis_packages' bbox?
        # }
        # df = geopandas.GeoDataFrame.from_features(feature_coll, crs=files['tracker']['trn/properties'][0])
        # df.to_file(hdf5_file_path + "package.gpkg", layer='countries', driver="GPKG")
'''


# # 0) read in params
#     parser = argparse.ArgumentParser(description='Sample preparation')
#     parser.add_argument('ParamFile', metavar='DIR',help='Path to training parameters stored in yaml')
#     args = parser.parse_args()
#     param_path = Path(args.ParamFile)
#     print(args.ParamFile)
#     params = read_params(args.ParamFile)

# 1) options
#-----------------------------------------------------------------------------------------------------------------------

hdf5_file_path = 'D:/NRCan_data/MECnet_implementation/runs/Hydro/samples256_overlap25_min-annot3_4bands_pls_work/'
checkpoint_file_path = ''
print('\n\n\n')



def read_params(param_file):
    yaml = YAML()
    yaml.preserve_quotes = True
    with open(param_file) as fp:
        data = yaml.load(fp)
        fp.close()
    return data

def plt_show(tracker_file, files):
    # TODO add h,w & ims_to_show as params & assert they have the same area!
    height, width = 3, 3
    fig = plt.figure()
    gs = fig.add_gridspec(height, width)
    axs = []
    y,x=0,0
    for N, pos in enumerate((40,41,42, 16,17,18, 0,1,2)):
        # print(tracker_file['trn/projection'][pos][0])
        w = tracker_file['trn/coords'][pos,0]
        e = tracker_file['trn/coords'][pos,1]
        s = tracker_file['trn/coords'][pos,2]
        n = tracker_file['trn/coords'][pos,3]
        axs.append(fig.add_subplot(gs[y, x]))
        axs[N].imshow(files['trn']['sat_img'][pos, ...])
        axs[N].set_title(str(pos))
        axs[N].set_xlabel(f'{w:.0f} | {e:.0f}')
        axs[N].set_ylabel(f'{s:.0f} | {n:.0f}')
        y = y+1 if y % (width-1) == 0 else 0
        x = x+1 if x % (height-1) == 0 else 0
    plt.show()

def run_model():
    # load checkpoint model and evaluate it
    model, model_name, criterion, optimizer, lr_scheduler = net(params, params['global']['num_classes']+1)
    checkpoint = load_checkpoint(checkpoint_file_path)
    model, _ = load_from_checkpoint(checkpoint, model)

    # if tst_dataloader:
    #     tst_report = evaluation(console, tracker,
    #                             eval_loader=tst_dataloader,
    #                             model=model,
    #                             criterion=criterion,
    #                             num_classes=num_classes_corrected,
    #                             batch_size=batch_size,
    #                             ep_idx=params['training']['num_epochs'],
    #                             progress_log=progress_log,
    #                             vis_params=params,
    #                             batch_metrics=params['training']['batch_metrics'],
    #                             dataset='tst',
    #                             device=device)
    #     tst_log.add_values(tst_report, params['training']['num_epochs'])
    #
    #     if bucket_name:
    #         bucket_filename = bucket_output_path.joinpath('last_epoch.pth.tar')
    #         bucket.upload_file("output.txt", bucket_output_path.joinpath(f"Logs/{now}_output.txt"))
    #         bucket.upload_file(filename, bucket_filename)

    # # for im in ('map_img', 'sat_img'):



try:
    tracker_file = h5py.File(hdf5_file_path + 'tracker' + '.hdf5', 'r')
    print('[bold white on green]tracker has OFFICIALLY been imported')
    files = {}

    for group_name in ['trn', 'tst', 'val']:
        files[group_name] = h5py.File(hdf5_file_path + group_name + '_samples.hdf5', 'r')
        print('[bold white on green]' + group_name + ' has OFFICIALLY been imported')
        # print()
        # print()
        # print(group_name+'/projection', '\t', tracker_file[group_name+'/projection'].shape,
        #                                       tracker_file[group_name+'/projection'].dtype,
        #                                       tracker_file[group_name+'/projection'].size)
        # print()
        # print(group_name+'/coords', '\t', tracker_file[group_name+'/coords'].shape,
        #                                   tracker_file[group_name+'/coords'].dtype,
        #                                   tracker_file[group_name+'/coords'].size)
    print('#'*35)



#                                                      PLOT
    plt_show(tracker_file, files)
############################################     EXPORT to GPKG     ####################################################

    features = {
        "type": "FeatureCollection",
        "features": [
                    # {
                    #     "id": "0",
                    #     "type": "Feature",
                    #     "properties": {"col1": "name1"},
                    #     "geometry": {"type": "Point",
                    #                  "coordinates": (1.0, 2.0)},
                    #     "bbox": (1.0, 2.0, 1.0, 2.0),
                    # }
                    ],
                }
    # print(tracker_file['trn'].keys())
    for grp in ['trn', 'tst', 'val']:
        for pos in range(0, tracker_file[grp+'/projection'].shape[0]):
            w = tracker_file[grp+'/coords'][pos,0]
            e = tracker_file[grp+'/coords'][pos,1]
            s = tracker_file[grp+'/coords'][pos,2]
            n = tracker_file[grp+'/coords'][pos,3]
            features['features'].append({
                                        "id": str(pos),
                                        "type": "Feature",
                                        "properties": {"sample_num": str(pos),
                                                       'dataset'   : grp},
                                        "geometry": {"type": "polygon",
                                                     "coordinates": (
                                                                     ((w,n),
                                                                      (w,s),
                                                                      (e,s),
                                                                      (e,n),
                                                                      (w,n)),
                                                                     # holes go on this line
                                                                    )
                                                     },
                                        "bbox": (w,e,n,s)
                                        })
    outline = geopandas.GeoDataFrame.from_features(features, crs=tracker_file['trn/projection'][pos][0]) # TODO make results per geopackage, mb just a txt file? also add in proj there too
    outline.to_file(hdf5_file_path + "vis.gpkg", layer='samp '+str(pos), driver="GPKG")







    print('[bold white on green]WE GOOOOOOOOOOOD')
    tracker_file.close()
    print('[bold white on red]tracker has OFFICIALLY been closed')
    for f in files.values():
        f.close()
        print('[bold white on red]' + group_name + ' has OFFICIALLY been closed')
except Exception as e:
    print('[bold white on red]ERRRRRRRRRRRRRRRRRRRRRRRRRRRRR')
    print(type(e))
    print(e)
    tracker_file.close()
    print('[bold white on red]tracker has OFFICIALLY been closed')
    for f in files.values():
        name = str(f.name)
        f.close()
        print('[bold white on red]' + name + ' has OFFICIALLY been closed')



'''
########################################################################################################################
        OLD hdf5 visualization tools
########################################################################################################################

# hdf5 OPTIONS :

data_location = '.\\run\\'

experiment_dir = 'samples256_overlap33_min-annot3_3bands_pls_work'
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

'''