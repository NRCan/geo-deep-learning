import h5py
from matplotlib import pyplot as plt
import numpy as np
from rich import inspect

# region 1) options

hdf5_file_path = 'D:/NRCan_data/MECnet_implementation/runs/Aerial/samples1024_overlap25_min-annot0_3bands_pls_work/'
things_to_track = ['perc_pad',
                   'iou',
                   'dist_from_edge',
                   'pxl_accuracy']#,'per_class_iou?'}
scaling_factor = 4

# endregion

for grp in ['val', 'trn']:
    with h5py.File(hdf5_file_path+'ims/'+grp+'_results.hdf5', 'r') as f:
        inspect(f['perc_pad'])
        fig = plt.figure(constrained_layout=True)
        plt.scatter(range(f['perc_pad'].shape[0]), f['perc_pad'])
        # fig.show()
        fig.savefig(hdf5_file_path+'ims/'+grp+'_perc_pad.png', bbox_inches='tight')