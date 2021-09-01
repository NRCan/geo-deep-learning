from typing import List, Sequence
import argparse, h5py, torch, geopandas
import time
from pathlib import Path
from ruamel_yaml import YAML
from models.model_choice import net, load_checkpoint, verify_weights
# from matplotlib import pyplot as plt
import numpy as np

from utils.utils import get_key_def, get_device_ids, gpu_stats
from utils.metrics import iou

def flatten_labels(annotations):
    """Flatten labels"""
    flatten = annotations.view(-1)
    return flatten


def flatten_outputs(predictions, number_of_classes):
    """Flatten the prediction batch except the prediction dimensions"""
    logits_permuted = predictions.permute(0, 2, 3, 1)
    logits_permuted_cont = logits_permuted.contiguous()
    outputs_flatten = logits_permuted_cont.view(-1, number_of_classes)
    return outputs_flatten
    # outputs_flatten = torch.tensor(predictions

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

    hdf5_file_path = 'D:/NRCan_data/MECnet_implementation/runs/Aerial/samples1024_overlap25_min-annot0_3bands_pls_work/'
    # checkpoint_file_path = ''
    things_to_track = ['perc_pad',
                       'iou']#,
                       # 'dist_from_edge',
                       # 'pxl_accuracy']#,'per_class_iou?'}
    scaling_factor = 4

    # torch.cuda.empty_cache()
    # endregion

    try: # incase any err, hdf5 NEED to be closed (or else they will be unusable)

    # region 2) h5 imports
    #     tracker_file = h5py.File(hdf5_file_path + 'tracker' + '.hdf5', 'r')
        print('tracker has OFFICIALLY been imported')
        files = {}

        for group_name in ['trn', 'tst', 'val']:
            files[group_name] = h5py.File(hdf5_file_path + group_name + '_samples.hdf5', 'r')
            print(group_name + ' has OFFICIALLY been imported')
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
    # #         exp_to_gpkg(tracker_file,
    # #                     hdf5_file_path)
    #
    #         features = {
    #                     "type": "FeatureCollection",
    #                     "features": [
    #                                 # {
    #                                 #     "id": "0",
    #                                 #     "type": "Feature",
    #                                 #     "properties": {"col1": "name1"},
    #                                 #     "geometry": {"type": "Point",
    #                                 #                  "coordinates": (1.0, 2.0)},
    #                                 #     "bbox": (1.0, 2.0, 1.0, 2.0),
    #                                 # }
    #                                 ],
    #                     }
    #         # print(tracker_file['trn'].keys())
    #         for grp in ['trn', 'tst', 'val']:
    #             for pos in range(0, tracker_file[grp+'/projection'].shape[0]):
    #                 w = tracker_file[grp+'/coords'][pos,0]
    #                 e = tracker_file[grp+'/coords'][pos,1]
    #                 s = tracker_file[grp+'/coords'][pos,2]
    #                 n = tracker_file[grp+'/coords'][pos,3]
    #                 features['features'].append({
    #                                             "id": str(pos),
    #                                             "type": "Feature",
    #                                             "properties": {"sample_num": str(pos),
    #                                                            'dataset'   : grp},
    #                                             "geometry": {"type": "polygon",
    #                                                          "coordinates": (
    #                                                                          ((w,n),
    #                                                                           (w,s),
    #                                                                           (e,s),
    #                                                                           (e,n),
    #                                                                           (w,n)),
    #                                                                          # any holes go on this line
    #                                                                         )
    #                                                          },
    #                                             "bbox": (w,e,n,s)
    #                                             })
    #                 outline = geopandas.GeoDataFrame.from_features(features, crs=tracker_file['trn/projection'][pos][0]) # TODO make results per geopackage, mb just a txt file? also add in proj there too
    #                 outline.to_file(hdf5_file_path + "vis.gpkg", layer='samp '+str(pos), driver="GPKG")
    # #                 features = {
    # #                             "type": "FeatureCollection",
    # #                             "features": [
    # #                                         # {
    # #                                         #     "id": "0",
    # #                                         #     "type": "Feature",
    # #                                         #     "properties": {"col1": "name1"},
    # #                                         #     "geometry": {"type": "Point",
    # #                                         #                  "coordinates": (1.0, 2.0)},
    # #                                         #     "bbox": (1.0, 2.0, 1.0, 2.0),
    # #                                         # }
    # #                                         ],
    # #                             }
    # #                 time.sleep(3)
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

    # region 5) run thru model(s)

        # region inits:
        sample_size = params['global']['samples_size']

        model_name = get_key_def('model_name', params['global'], expected_type=str).lower()
        batch_size = params['training']['batch_size']
        num_devices = params['global']['num_gpus']
        # gpu parameters
        num_devices = get_key_def('num_gpus', params['global'], default=0, expected_type=int)
        if num_devices and not num_devices >= 0:
            raise ValueError("missing mandatory num gpus parameter")
        default_max_used_ram = 15
        max_used_ram = get_key_def('max_used_ram', params['global'], default=default_max_used_ram, expected_type=int)
        max_used_perc = get_key_def('max_used_perc', params['global'], default=15, expected_type=int)
        num_workers = num_devices * 4 if num_devices > 1 else 4

        gpu_devices_dict = get_device_ids(num_devices,
                                          max_used_ram_perc=max_used_ram,
                                          max_used_perc=max_used_perc)
        print(f'GPUs devices available: {gpu_devices_dict}')
        num_devices = len(gpu_devices_dict.keys())
        device = torch.device(f'cuda:{list(gpu_devices_dict.keys())[0]}' if gpu_devices_dict else 'cpu')
        # model params
        loss_fn = get_key_def('loss_fn', params['training'], default='CrossEntropy', expected_type=str)
        class_weights = get_key_def('class_weights', params['training'], default=None, expected_type=Sequence)
        if class_weights:
            verify_weights(params['global']['num_classes'], class_weights)
        optimizer = get_key_def('optimizer', params['training'], default='adam', expected_type=str)
        pretrained = get_key_def('pretrained', params['training'], default=True, expected_type=bool)
        train_state_dict_path = get_key_def('state_dict_path', params['training'], default=None, expected_type=str)
        if train_state_dict_path and not Path(train_state_dict_path).is_file():
            raise FileNotFoundError(f'Could not locate pretrained checkpoint for training: {train_state_dict_path}')
        dropout_prob = get_key_def('dropout_prob', params['training'], default=None, expected_type=float)
        # Read the concatenation point
        # TODO: find a way to maybe implement it in classification one day
        conc_point = get_key_def('concatenate_depth', params['global'], None)
        # coordconv parameters
        coordconv_params = {}
        for param, val in params['global'].items():
            if 'coordconv' in param:
                coordconv_params[param] = val

        dontcare_val = get_key_def("ignore_index", params["training"], -1)
        num_bands = params['global']['number_of_bands']
        num_classes_corrected = params['global']['num_classes']+1

        model, model_name, criterion, optimizer, lr_scheduler = net(model_name=model_name,
                                                                    num_bands=num_bands,
                                                                    num_channels=num_classes_corrected,
                                                                    dontcare_val=dontcare_val,
                                                                    num_devices=num_devices,
                                                                    train_state_dict_path=train_state_dict_path,
                                                                    pretrained=pretrained,
                                                                    dropout_prob=dropout_prob,
                                                                    loss_fn=loss_fn,
                                                                    class_weights=class_weights,
                                                                    optimizer=optimizer,
                                                                    net_params=params,
                                                                    conc_point=conc_point,
                                                                    coordconv_params=coordconv_params)

    # endregion
        model.eval()
        for grp in ['val', 'tst', 'trn']:

            batches = files[grp]['sat_img'].shape[0] // batch_size
            for i in range(batches):
                # region UGH
                # res, mem = gpu_stats(device=device.index)
                # print('\t\t\t\t\t', f'{res.gpu}%\t', f'{mem.used / (1024 ** 2):.0f}/{mem.total / (1024 ** 2):.0f} MiB')
                # torch.cuda.empty_cache()
                # res, mem = gpu_stats(device=device.index)
                # print('\t\t\t\t\t', f'{res.gpu}%\t', f'{mem.used / (1024 ** 2):.0f}/{mem.total / (1024 ** 2):.0f} MiB')
                # optimizer = get_key_def('optimizer', params['training'], default='adam', expected_type=str)
                # model, model_name, criterion, optimizer, lr_scheduler = net(model_name=model_name,
                #                                                 num_bands=num_bands,
                #                                                 num_channels=num_classes_corrected,
                #                                                 dontcare_val=dontcare_val,
                #                                                 num_devices=num_devices,
                #                                                 train_state_dict_path=train_state_dict_path,
                #                                                 pretrained=pretrained,
                #                                                 dropout_prob=dropout_prob,
                #                                                 loss_fn=loss_fn,
                #                                                 class_weights=class_weights,
                #                                                 optimizer=optimizer,
                #                                                 net_params=params,
                #                                                 conc_point=conc_point,
                #                                                 coordconv_params=coordconv_params)
                # res, mem = gpu_stats(device=device.index)
                # print('\t\t\t\t\t', f'{res.gpu}%\t', f'{mem.used / (1024 ** 2):.0f}/{mem.total / (1024 ** 2):.0f} MiB')
                # endregion
                print(grp, i*batch_size,'/',batches, end=' - ')
                batch = np.arange((i*batch_size), (i*batch_size) + batch_size, dtype=np.uint8)
                input = files['trn']['sat_img'][batch].astype(np.float32)
                input = torch.tensor(input, dtype=torch.float32)
                input = input.permute(0, 3, 1, 2)
                input = input.to(device)
                output = model(input)
                output.detach().cpu().numpy()
                res, mem = gpu_stats(device=device.index)
                print('\t\t\t', f'{res.gpu}%', f'{mem.used / (1024 ** 2):.0f}/{mem.total / (1024 ** 2):.0f} MiB')
                labels = files[grp]['map_img'][batch]

                # output = files[grp]['sat_img'][batch]


                # fig = plt.figure(figsize=(20,20*batch_size), constrained_layout=True)
                #
                # axs = {}
                # ax_count = 0
                # SAVE = False
                with h5py.File(hdf5_file_path+'ims/'+grp+'_results.hdf5', 'w') as f:
                    for thing in things_to_track:
                        if thing not in f.keys():
                            f.create_dataset(thing, (files[grp]['map_img'].shape[0], 1))
                            print(f[thing].shape[0])
                    for j in range(batch_size):
                        indexes = np.where(labels[j, ...]==dontcare_val)
                        # f['perc_pad'].resize(f['perc_pad'].shape[0]+1, axis=0)
                        for thing in things_to_track:
                            if thing=='perc_pad':
                                f[thing][i+j, ...] = len(indexes[0]) / (sample_size*sample_size)
                            if thing=='iou':
                                labels_flatten = flatten_labels(labels)
                                outputs_flatten = flatten_outputs(output, num_classes_corrected)
                                a, segmentation = torch.max(outputs_flatten, dim=1)
                                eval_metrics = iou(segmentation, labels_flatten, batch_size, num_classes_corrected, eval_metrics)
                                # for u in np.unique(labels):
                                #     if u==
                                f[thing][i+j, ...] = eval_metrics["loss"].avg
                    print('done!')

    # endregion

    # region X) close hdf5 files
        print('WE GOOOOOOOOOOOD')
        # tracker_file.close()
        print('tracker has OFFICIALLY been closed')
        for f in files:
            files[f].close()
            print('' + f + ' has OFFICIALLY been closed')
    except Exception as e:
        print('ERRRRRRRRRRRRRRRRRRRRRRRRRRRRR')
        print(type(e))
        print(e)
        # tracker_file.close()
        print('tracker has OFFICIALLY been closed')
        for f in files:
            files[f].close()
            print('' + f + ' has OFFICIALLY been closed')