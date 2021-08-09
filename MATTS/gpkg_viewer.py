import argparse, h5py, torch, geopandas
from pathlib import Path
from ruamel_yaml import YAML
from models.model_choice import net, load_checkpoint
from matplotlib import pyplot as plt
import numpy as np
# import cv2
# from torchsummary import summary as torch_summary

from torchvision import transforms
from utils import augmentation as aug, create_dataset
from torch.utils.data import DataLoader
from utils.utils import load_from_checkpoint, get_key_def, get_device_ids
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
    writer = SummaryWriter('D:/NRCan_data/runs')

    hdf5_file_path = 'D:/NRCan_data/MECnet_implementation/runs/Hydro/samples256_overlap25_min-annot3_4bands_pls_work/'
    checkpoint_file_path = ''
    print('\n\n\n')
    # endregion

    try: # incase any err, hdf5 NEED to be closed (or else they will be unusable)
# region 2) imports
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
        exp_to_gpkg(tracker_file,
                    hdf5_file_path)
    # endregion

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

# region 5) run thru model(s)
    # inits:
        batch_size = params['training']['batch_size']
        num_devices = params['global']['num_gpus']
        # # list of GPU devices that are available and unused. If no GPUs, returns empty list
        max_used_ram = get_key_def('max_used_ram', params['global'], 2000, expected_type=int)
        max_used_perc = get_key_def('max_used_perc', params['global'], 15, expected_type=int)
        lst_device_ids = get_device_ids(
            num_devices, max_used_ram=max_used_ram, max_used_perc=max_used_perc, debug=False) \
            if torch.cuda.is_available() else []
        num_devices = len(lst_device_ids) if lst_device_ids else 0
        device = torch.device(f'cuda:{lst_device_ids[0]}' if torch.cuda.is_available() and lst_device_ids else 'cpu')
        # console.print(device, style='bold #FFFFFF on green', justify="center")
        num_workers = num_devices * 4 if num_devices > 1 else 4

        dontcare_val = get_key_def("ignore_index", params["training"], -1)
        num_bands = params['global']['number_of_bands']
        num_classes_corrected = params['global']['num_classes']+1

    # create data loaders:
    #     height, width = 4, 3
    #     fig = plt.figure()
    #     gs = fig.add_gridspec(height, width)
    #     axs = []
    #     y,x=0,0
        ims_to_use = {'trn' : [0,1,2],
                      'tst' : [],
                      'val' : [0,1,2]}
        data = []
        for grp in ims_to_use:
            for imN in ims_to_use[grp]:
                # aug.compose_transforms(params, subset, type='radiometric'), #FIXME
                # aug.compose_transforms(params, subset, type='geometric', ignore_index=dontcare_val),

                # datasets[subset] = create_dataset.SegmentationDataset(hdf5_file_path, subset, num_bands, # TODO: add option to work with meta_map
                #                                                       max_sample_count=samples.shape[0],
                #                                                       dontcare=dontcare_val,
                #                                                       radiom_transform=None,#aug.compose_transforms(params, subset, type='radiometric'), #FIXME
                #                                                       geom_transform=None,#aug.compose_transforms(params, subset, type='geometric', ignore_index=dontcare_val),
                #                                                       totensor_transform=aug.compose_transforms(params, subset, type='totensor'),
                #                                                       params=params,
                #                                                       debug=False)

                # aug.compose_transforms(params, subset, type='totensor'),
                # transforms.Compose(lst_trans)

                # convert to TENSORs
                sat_img = np.nan_to_num(files[grp]['sat_img'][imN], copy=False)
                sat_img = np.float32(np.transpose(sat_img, (2, 0, 1)))
                sat_img = torch.from_numpy(sat_img)
                data.append({'sat_img': sat_img,
                             'map_img': torch.from_numpy(np.int64(files[grp]['map_img'][imN]))})

                writer.add_image('sat_img ' + grp + str(imN), data[-1]['sat_img'], 0)

                # TODO: map_img not exporting to tensorboard with new res values TIP: look at method in GDL vis process
                # map_img = torch.zeros((3, data[-1]['map_img'].shape[0], data[-1]['map_img'].shape[1]), dtype=torch.uint8)
                # vals = torch.unique(data[-1]['map_img'])
                # new_vals = [(v * (255 // (len(vals)-1))) for v in range(0,len(vals))]
                # map_img2 = torch.zeros((3, data[-1]['map_img'].shape[0], data[-1]['map_img'].shape[1]), dtype=torch.uint8)
                #
                # # map_img2[vals] = new_vals
                # writer.add_image('map_img2 ' + grp + str(imN), map_img2, 0)
                # for v in range(len(vals)):
                #     map_img[torch.where(map_img==int(vals[v]))] = int(new_vals[v])
                # writer.add_image('map_img ' + grp + str(imN), map_img, 0)



                # if grp == 'trn':
                #     axs.append(fig.add_subplot(gs[0, imN]))
                #     axs[-1].imshow(np.transpose(sat_img.numpy(), (1, 2, 0)))
                #     # axs[-1].set_title()
                #     axs.append(fig.add_subplot(gs[1, imN]))
                #     map_img = data[-1]['map_img']
                #     # classN = len(np.unique(map_img))
                #     # for uniq in np.unique(map_img):
                #     print(np.unique(map_img))
                #         # map_img[np.where(map_img==uniq)] = uniq // 255
                #     # axs[-1].imshow()
                #     # axs[-1].set_title()
                # elif grp == 'val':
                #     axs.append(fig.add_subplot(gs[0, imN]))
                #     axs[-1].imshow(sat_img.numpy)
        # plt.show()

                # dataloader = DataLoader(data,
                #                         batch_size=4,#batch_size,
                #                         num_workers=num_workers,
                #                         sampler=[0,1,2,3],
                #                         drop_last=True)


    # load checkpoint model:
        model, model_name, criterion, optimizer, lr_scheduler = net(params, num_classes_corrected)
        if not checkpoint_file_path == '':
            checkpoint = load_checkpoint(checkpoint_file_path)
            model, _ = load_from_checkpoint(checkpoint, model)
        # input =
        # writer.add_graph(model, torch.randn(64, 3, 3, 3).to(device), verbose=True)
        # writer.add_graph(model, sat_img, verbose=True)
        writer.close()

    # # run!
    #     with Tracking_Pane(console=console, mode='train') as tracker: # , other_renderables=[experiment_table]     # task_ids: 0=epoch, 1=trn batch, 2=vis batch (ie. order they are create in utils.tracker)
    #
    #         report = evaluation(console, tracker,
    #                                 eval_loader=dataloader,
    #                                 model=model,
    #                                 criterion=criterion,
    #                                 num_classes=num_classes_corrected,
    #                                 batch_size=batch_size,
    #                                 ep_idx=params['training']['num_epochs'],
    #                                 progress_log=None,
    #                                 vis_params=params,
    #                                 batch_metrics=params['training']['batch_metrics'],
    #                                 dataset='tst',
    #                                 device=device)

        # if bucket_name:
        #     bucket_filename = bucket_output_path.joinpath('last_epoch.pth.tar')
        #     bucket.upload_file("output.txt", bucket_output_path.joinpath(f"Logs/{now}_output.txt"))
        #     bucket.upload_file(filename, bucket_filename)

    # endregion

    # region X) close hdf5 files
        print('[bold white on green]WE GOOOOOOOOOOOD')
        tracker_file.close()
        print('[bold white on green]tracker has OFFICIALLY been closed')
        for f in files:
            files[f].close()
            print('[bold white on green]' + f + ' has OFFICIALLY been closed')
    except Exception as e:
        print('[bold white on red]ERRRRRRRRRRRRRRRRRRRRRRRRRRRRR')
        print(type(e))
        print(e)
        tracker_file.close()
        print('[bold white on red]tracker has OFFICIALLY been closed')
        for f in files:
            files[f].close()
            print('[bold white on red]' + f + ' has OFFICIALLY been closed')



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