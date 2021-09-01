from typing import List, Sequence
import argparse, h5py, torch, geopandas
import time
from pathlib import Path
from ruamel_yaml import YAML
from models.model_choice import net, load_checkpoint, verify_weights
from matplotlib import pyplot as plt
import numpy as np
# import cv2
# from torchsummary import summary as torch_summary

from torchvision import transforms
from utils import augmentation as aug, create_dataset
from torch.utils.data import DataLoader
from utils.utils import load_from_checkpoint, get_key_def, get_device_ids, gpu_stats
from train_segmentation import evaluation

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from utils.tracker import Tracking_Pane
from rich.console import Console
from rich import print, inspect
console = Console()

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
def exp_to_gpkg(tracker_file,
                hdf5_file_path):
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






#-----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':


# region 0) read in params
    def read_params(param_file):
        yaml = YAML()
        yaml.preserve_quotes = True
        with open(param_file) as fp:
            data = yaml.load(fp)
            fp.close()
        return data

    parser = argparse.ArgumentParser(description='Sample preparation')
    parser.add_argument('ParamFile', metavar='DIR',help='Path to training parameters stored in yaml')
    args = parser.parse_args()
    param_path = Path(args.ParamFile)
    print(args.ParamFile)
    params = read_params(args.ParamFile)
    # endregion

# region 1) options
#     writer = SummaryWriter('D:/NRCan_data/runs')

    hdf5_file_path = 'D:/NRCan_data/MECnet_implementation/runs/Aerial/samples1024_overlap0_min-annot0_3bands_pls_work/'
    # checkpoint_file_path = ''
    torch.cuda.empty_cache()
    # endregion

    # try: # incase any err, hdf5 NEED to be closed (or else they will be unusable)

# region 2) h5 imports
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
# endregion

# region 3) to GPKG
#         exp_to_gpkg(tracker_file,
#                     hdf5_file_path)

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
            features['features'].append({"id": str(pos),
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
                                                                      # any holes go on this line
                                                                     )
                                                      },
                                         "bbox": (w,e,n,s)
                                         })
            outline = geopandas.GeoDataFrame.from_features(features, crs=tracker_file['trn/projection'][pos][0]) # TODO make results per geopackage, mb just a txt file? also add in proj there too
            outline.to_file(hdf5_file_path + "vis.gpkg", layer='outputs' , driver="GPKG")
#                 features = {
#                             "type": "FeatureCollection",
#                             "features": [
#                                         # {
#                                         #     "id": "0",
#                                         #     "type": "Feature",
#                                         #     "properties": {"col1": "name1"},
#                                         #     "geometry": {"type": "Point",
#                                         #                  "coordinates": (1.0, 2.0)},
#                                         #     "bbox": (1.0, 2.0, 1.0, 2.0),
#                                         # }
#                                         ],
#                             }
#                 time.sleep(3)
#     # endregion

# region 4) matplotlib SHOW
#     # TODO add h,w & ims_to_show as params & assert they have the same area!
#     height, width = 3, 3
#     fig = plt.figure()
#     gs = fig.add_gridspec(height, width)
#     axs = []
#     y,x=0,0
#     for N, pos in enumerate((40,41,42, 16,17,18, 0,1,2)):
#         # print(tracker_file['trn/projection'][pos][0])
#         w = tracker_file['trn/coords'][pos,0]
#         e = tracker_file['trn/coords'][pos,1]
#         s = tracker_file['trn/coords'][pos,2]
#         n = tracker_file['trn/coords'][pos,3]
#         axs.append(fig.add_subplot(gs[y, x]))
#         axs[N].imshow(files['trn']['sat_img'][pos, ...])
#         axs[N].set_title(str(pos))
#         axs[N].set_xlabel(f'{w:.0f} | {e:.0f}')
#         axs[N].set_ylabel(f'{s:.0f} | {n:.0f}')
#         y = y+1 if y % (width-1) == 0 else 0
#         x = x+1 if x % (height-1) == 0 else 0
#     plt.show()
# endregion

# # region 5) run thru model(s)
#
#     # region inits:
#     model_name = get_key_def('model_name', params['global'], expected_type=str).lower()
#     batch_size = params['training']['batch_size']
#     num_devices = params['global']['num_gpus']
#     # gpu parameters
#     num_devices = get_key_def('num_gpus', params['global'], default=0, expected_type=int)
#     if num_devices and not num_devices >= 0:
#         raise ValueError("missing mandatory num gpus parameter")
#     default_max_used_ram = 15
#     max_used_ram = get_key_def('max_used_ram', params['global'], default=default_max_used_ram, expected_type=int)
#     max_used_perc = get_key_def('max_used_perc', params['global'], default=15, expected_type=int)
#     num_workers = num_devices * 4 if num_devices > 1 else 4
#
#     gpu_devices_dict = get_device_ids(num_devices,
#                                       max_used_ram_perc=max_used_ram,
#                                       max_used_perc=max_used_perc)
#     print(f'GPUs devices available: {gpu_devices_dict}')
#     num_devices = len(gpu_devices_dict.keys())
#     device = torch.device(f'cuda:{list(gpu_devices_dict.keys())[0]}' if gpu_devices_dict else 'cpu')
#     # model params
#     loss_fn = get_key_def('loss_fn', params['training'], default='CrossEntropy', expected_type=str)
#     class_weights = get_key_def('class_weights', params['training'], default=None, expected_type=Sequence)
#     if class_weights:
#         verify_weights(params['global']['num_classes'], class_weights)
#     optimizer = get_key_def('optimizer', params['training'], default='adam', expected_type=str)
#     pretrained = get_key_def('pretrained', params['training'], default=True, expected_type=bool)
#     train_state_dict_path = get_key_def('state_dict_path', params['training'], default=None, expected_type=str)
#     if train_state_dict_path and not Path(train_state_dict_path).is_file():
#         raise FileNotFoundError(f'Could not locate pretrained checkpoint for training: {train_state_dict_path}')
#     dropout_prob = get_key_def('dropout_prob', params['training'], default=None, expected_type=float)
#     # Read the concatenation point
#     # TODO: find a way to maybe implement it in classification one day
#     conc_point = get_key_def('concatenate_depth', params['global'], None)
#     # coordconv parameters
#     coordconv_params = {}
#     for param, val in params['global'].items():
#         if 'coordconv' in param:
#             coordconv_params[param] = val
#
#     dontcare_val = get_key_def("ignore_index", params["training"], -1)
#     num_bands = params['global']['number_of_bands']
#     num_classes_corrected = params['global']['num_classes']+1
#
#     model, model_name, criterion, optimizer, lr_scheduler = net(model_name=model_name,
#                                                                 num_bands=num_bands,
#                                                                 num_channels=num_classes_corrected,
#                                                                 dontcare_val=dontcare_val,
#                                                                 num_devices=num_devices,
#                                                                 train_state_dict_path=train_state_dict_path,
#                                                                 pretrained=pretrained,
#                                                                 dropout_prob=dropout_prob,
#                                                                 loss_fn=loss_fn,
#                                                                 class_weights=class_weights,
#                                                                 optimizer=optimizer,
#                                                                 net_params=params,
#                                                                 conc_point=conc_point,
#                                                                 coordconv_params=coordconv_params)
#
# # endregion
#
#     sample_size = params['global']['samples_size']
#
#     for grp in ['trn', 'tst', 'val']:
#
#
#         batches = files[grp]['sat_img'].shape[0] // batch_size
#         for i in range(batches):
#             # region UGH
#             res, mem = gpu_stats(device=device.index)
#             print('\t\t\t\t\t', f'{res.gpu}%\t', f'{mem.used / (1024 ** 2):.0f}/{mem.total / (1024 ** 2):.0f} MiB')
#             torch.cuda.empty_cache()
#             res, mem = gpu_stats(device=device.index)
#             print('\t\t\t\t\t', f'{res.gpu}%\t', f'{mem.used / (1024 ** 2):.0f}/{mem.total / (1024 ** 2):.0f} MiB')
#             optimizer = get_key_def('optimizer', params['training'], default='adam', expected_type=str)
#             model, model_name, criterion, optimizer, lr_scheduler = net(model_name=model_name,
#                                                             num_bands=num_bands,
#                                                             num_channels=num_classes_corrected,
#                                                             dontcare_val=dontcare_val,
#                                                             num_devices=num_devices,
#                                                             train_state_dict_path=train_state_dict_path,
#                                                             pretrained=pretrained,
#                                                             dropout_prob=dropout_prob,
#                                                             loss_fn=loss_fn,
#                                                             class_weights=class_weights,
#                                                             optimizer=optimizer,
#                                                             net_params=params,
#                                                             conc_point=conc_point,
#                                                             coordconv_params=coordconv_params)
#             res, mem = gpu_stats(device=device.index)
#             print('\t\t\t\t\t', f'{res.gpu}%\t', f'{mem.used / (1024 ** 2):.0f}/{mem.total / (1024 ** 2):.0f} MiB')
#             # endregion
#             print(i*batch_size, grp)
#             batch = np.arange((i*batch_size), (i*batch_size) + batch_size, dtype=np.uint8)
#             input = files['trn']['sat_img'][batch].astype(np.float32)
#             input = torch.tensor(input, dtype=torch.float32)
#             input = input.permute(0, 3, 1, 2)
#             input = input.to(device)
#             output = model(input)
#             res, mem = gpu_stats(device=device.index)
#             print('\t\t\t', f'{res.gpu}%\y', f'{mem.used / (1024 ** 2):.0f}/{mem.total / (1024 ** 2):.0f} MiB')
#             output.cpu()
#             res, mem = gpu_stats(device=device.index)
#             print('\t\t\t', f'{res.gpu}%\t', f'{mem.used / (1024 ** 2):.0f}/{mem.total / (1024 ** 2):.0f} MiB')
#             # output = files[grp]['map_img'][batch]
#             labels = files[grp]['map_img'][batch]
#
#
#             # fig = plt.figure(figsize=(20,20*batch_size), constrained_layout=True)
#             #
#             # axs = {}
#             # ax_count = 0
#             # SAVE = False
#             with h5py.File(hdf5_file_path+'ims/results.hdf5', 'w') as f:
#                 if 'perc_pad' not in f.keys():
#                     f.create_dataset('perc_pad', (0, 1), maxshape=(None, 1))
#                 for j in range(batch_size):
#                     indexes = np.where(labels[j, ...]==dontcare_val)
#                     # perc_pad = len(indexes[0]) / (sample_size*sample_size)
#                     f['perc_pad'].resize(f['perc_pad'].shape[0]+1, axis=0)
#                     f['perc_pad'][f['perc_pad'].shape[0]-1, ...] = len(indexes[0]) / (sample_size*sample_size)
#
#     with h5py.File(hdf5_file_path+'ims/results.hdf5', 'r') as f:
#         fig = plt.figure(constrained_layout=True)
#         fig.plot(range(f['perc_pad'].shape[0]), f['perec_pad'])
#         fig.show()
#                 # if len(indexes[0]) > 0:
#                 #     print(grp, batch[j], str(perc_pad * 100)+'%')
#                     # fig = plt.figure(constrained_layout=True)
#                     # gs = fig.add_gridspec(4, 1)
#                     # ax1 = fig.add_subplot(gs[0, 0])
#                     # ax1.imshow(output[j, ...])
#                     # ax1 = fig.add_subplot(gs[2, 0])
#                     # ax1.imshow(files[grp]['map_img'][batch[j]])
#                     # ax1.set_title(str(perc_pad * 100)+'% '+grp+str(batch[j]))
#                     # ax2 = fig.add_subplot(gs[1, 0])
#                     # ax2.imshow(files[grp]['sat_img'][batch[j]])
#                     # fig.savefig(hdf5_file_path+'ims/pad-'+grp+str(batch[j])+'.png', bbox_inches='tight')
#                     # plt.clf()
#             #
#             #
#             #         axs[j] = fig.add_subplot(gs[0, ax_count])
#             #
#             #         axs[j].imshow(files[grp]['sat_img'][batch[j]])
#             #         axs[j].imshow(output[j, ...])
#             #         axs[j].set_title(str(len(indexes[0]) / (sample_size*sample_size))+' '+grp+str(batch[j]))
#             #         axs[j].set_yticks([])
#             #         axs[j].set_xticks([])
#             #         ax_count += 1
#             #         SAVE = True
#             # if SAVE:
#
#             # if i > 40:
#             #     break
#
#
#
#
#
#
#
# #     # create data loaders:
# #     #     height, width = 4, 3
# #     #     fig = plt.figure()
# #     #     gs = fig.add_gridspec(height, width)
# #     #     axs = []
# #     #     y,x=0,0
# #
# #
# #
# #         # ims_to_use = {'trn' : [0,1,2],
# #         #               'tst' : [],
# #         #               'val' : [0,1,2]}
# #         # data = []
# #         # for grp in ims_to_use:
# #         #     for imN in ims_to_use[grp]:
# #         #         # aug.compose_transforms(params, subset, type='radiometric'), #FIXME
# #         #         # aug.compose_transforms(params, subset, type='geometric', ignore_index=dontcare_val),
# #         #
# #         #         # datasets[subset] = create_dataset.SegmentationDataset(hdf5_file_path, subset, num_bands, # TODO: add option to work with meta_map
# #         #         #                                                       max_sample_count=samples.shape[0],
# #         #         #                                                       dontcare=dontcare_val,
# #         #         #                                                       radiom_transform=None,#aug.compose_transforms(params, subset, type='radiometric'), #FIXME
# #         #         #                                                       geom_transform=None,#aug.compose_transforms(params, subset, type='geometric', ignore_index=dontcare_val),
# #         #         #                                                       totensor_transform=aug.compose_transforms(params, subset, type='totensor'),
# #         #         #                                                       params=params,
# #         #         #                                                       debug=False)
# #         #
# #         #         # aug.compose_transforms(params, subset, type='totensor'),
# #         #         # transforms.Compose(lst_trans)
# #         #
# #         #         # convert to TENSORs
# #         #         sat_img = np.nan_to_num(files[grp]['sat_img'][imN], copy=False)
# #         #         sat_img = np.float32(np.transpose(sat_img, (2, 0, 1)))
# #         #         sat_img = torch.from_numpy(sat_img)
# #         #         data.append({'sat_img': sat_img,
# #         #                      'map_img': torch.from_numpy(np.int64(files[grp]['map_img'][imN]))})
# #         #
# #         #         # writer.add_image('sat_img ' + grp + str(imN), data[-1]['sat_img'], 0)
# #
# #                 # TODO: map_img not exporting to tensorboard with new res values TIP: look at method in GDL vis process
# #                 # map_img = torch.zeros((3, data[-1]['map_img'].shape[0], data[-1]['map_img'].shape[1]), dtype=torch.uint8)
# #                 # vals = torch.unique(data[-1]['map_img'])
# #                 # new_vals = [(v * (255 // (len(vals)-1))) for v in range(0,len(vals))]
# #                 # map_img2 = torch.zeros((3, data[-1]['map_img'].shape[0], data[-1]['map_img'].shape[1]), dtype=torch.uint8)
# #                 #
# #                 # # map_img2[vals] = new_vals
# #                 # writer.add_image('map_img2 ' + grp + str(imN), map_img2, 0)
# #                 # for v in range(len(vals)):
# #                 #     map_img[torch.where(map_img==int(vals[v]))] = int(new_vals[v])
# #                 # writer.add_image('map_img ' + grp + str(imN), map_img, 0)
# #
# #
# #
# #                 # if grp == 'trn':
# #                 #     axs.append(fig.add_subplot(gs[0, imN]))
# #                 #     axs[-1].imshow(np.transpose(sat_img.numpy(), (1, 2, 0)))
# #                 #     # axs[-1].set_title()
# #                 #     axs.append(fig.add_subplot(gs[1, imN]))
# #                 #     map_img = data[-1]['map_img']
# #                 #     # classN = len(np.unique(map_img))
# #                 #     # for uniq in np.unique(map_img):
# #                 #     print(np.unique(map_img))
# #                 #         # map_img[np.where(map_img==uniq)] = uniq // 255
# #                 #     # axs[-1].imshow()
# #                 #     # axs[-1].set_title()
# #                 # elif grp == 'val':
# #                 #     axs.append(fig.add_subplot(gs[0, imN]))
# #                 #     axs[-1].imshow(sat_img.numpy)
# #         # plt.show()
# #
# #                 # dataloader = DataLoader(data,
# #                 #                         batch_size=4,#batch_size,
# #                 #                         num_workers=num_workers,
# #                 #                         sampler=[0,1,2,3],
# #                 #                         drop_last=True)
# #
# #
# #         # writer.add_graph(model, sat_img, verbose=True)
# #         # writer.close()
#
# # endregion

# region X) close hdf5 files
#     print('[bold white on green]WE GOOOOOOOOOOOD')
    tracker_file.close()
    print('[bold white on green]tracker has OFFICIALLY been closed')
    for f in files:
        files[f].close()
        print('[bold white on green]' + f + ' has OFFICIALLY been closed')
    # except Exception as e:
    #     print('[bold white on red]ERRRRRRRRRRRRRRRRRRRRRRRRRRRRR')
    #     print(type(e))
    #     print(e)
    #     tracker_file.close()
    #     print('[bold white on red]tracker has OFFICIALLY been closed')
    #     for f in files:
    #         files[f].close()
    #         print('[bold white on red]' + f + ' has OFFICIALLY been closed')



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