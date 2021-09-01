# import argparse, torch
# from pathlib import Path
# from ruamel_yaml import YAML
# from models.model_choice import net, load_checkpoint
# from torch.utils.tensorboard import SummaryWriter
# from utils.utils import load_from_checkpoint, get_key_def, get_device_ids
# # import cupy
# checkpoint_file_path = ''
# writer = SummaryWriter('D:/NRCan_data/runs')
#
# # region 0) read in params
# def read_params(param_file):
#     yaml = YAML()
#     yaml.preserve_quotes = True
#     with open(param_file) as fp:
#         data = yaml.load(fp)
#         fp.close()
#     return data
#
# parser = argparse.ArgumentParser(description='Sample preparation')
# parser.add_argument('ParamFile', metavar='DIR',help='Path to training parameters stored in yaml')
# args = parser.parse_args()
# param_path = Path(args.ParamFile)
# print(args.ParamFile)
# params = read_params(args.ParamFile)
# # endregion
#
# # region inits
# batch_size = params['training']['batch_size']
# num_devices = params['global']['num_gpus']
# # # list of GPU devices that are available and unused. If no GPUs, returns empty list
# max_used_ram = get_key_def('max_used_ram', params['global'], default=default_max_used_ram, expected_type=int)
# max_used_perc = get_key_def('max_used_perc', params['global'], default=15, expected_type=int)
# num_workers = num_devices * 4 if num_devices > 1 else 4
#
# gpu_devices_dict = get_device_ids(num_devices,
#                                   max_used_ram_perc=max_used_ram,
#                                   max_used_perc=max_used_perc)
# num_devices = len(lst_device_ids) if lst_device_ids else 0
# device = torch.device(f'cuda:{lst_device_ids[0]}' if torch.cuda.is_available() and lst_device_ids else 'cpu')
# # console.print(device, style='bold #FFFFFF on green', justify="center")
# num_workers = num_devices * 4 if num_devices > 1 else 4
#
# dontcare_val = get_key_def("ignore_index", params["training"], -1)
# num_bands = params['global']['number_of_bands']
# num_classes_corrected = params['global']['num_classes']+1
# # endregion
#
#
#
#
#
# params['global']['model_name'] = 'mecnet'
#
# # load checkpoint model:
# model, model_name, criterion, optimizer, lr_scheduler = net(params, num_classes_corrected)
# if not checkpoint_file_path == '':
#     checkpoint = load_checkpoint(checkpoint_file_path)
#     model, _ = load_from_checkpoint(checkpoint, model)
# # input =
# writer.add_graph(model, torch.randn(1,3,512,512).to(device))#, verbose=True)
# writer.close()

class hello:
    def __init__(self):
        print('heyo from PARENT')
    def gett(self):
        print('\t\tworked!')

class test(hello):
    def __init__(self):
        super().__init__()
        print('heyo from CHILD')
    def gett(self):
        print('...')
        super().gett()

t = test()
t.gett()