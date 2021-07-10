import argparse, io, sys, os, h5py
from pathlib import Path
from ruamel_yaml import YAML
import csv

from PIL import Image
import numpy as np

from rich.console import Console, RenderGroup
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.tree import Tree
from rich.columns import Columns
from rich import box
from torchsummary import summary as torch_summary

# from ray import tune
# from ray.tune import CLIReporter
# from ray.tune.schedulers import ASHAScheduler

# from utils.tracker import Tracker

from models.model_choice import net

from images_to_samples import main as IM_TO_SAMPLES_main
from train_segmentation import main as TRAIN_main


def make_csv_trckr(csv_filename):
    """
    Open csv file and parse it, returning a list of dict.
    - tif full path
    - metadata yml full path (may be empty string if unavailable)
    - gpkg full path
    - attribute_name
    - dataset (trn or tst)
    """
    list_values = []
    with open(csv_filename, 'r') as f:
        reader = csv.reader(f)
        for index, row in enumerate(reader):
            row_length = len(row) if index == 0 else row_length
            assert len(row) == row_length, "Rows in csv should be of same length"
            row.extend([None] * (5 - len(row)))  # fill row with None values to obtain row of length == 5

            list_values.append({'tif': row[0], 'meta': row[1], 'gpkg': row[2], 'attribute_name': row[3], 'dataset': row[4]})
            assert Path(row[0]).is_file(), f'Tif raster not found "{row[0]}"'
            if row[2]:
                assert Path(row[2]).is_file(), f'Gpkg not found "{row[2]}"'
                assert isinstance(row[3], str)
    try:
        # Try sorting according to dataset name (i.e. group "train", "val" and "test" rows together)
        list_values = sorted(list_values, key=lambda k: k['dataset'])
    except TypeError:
        list_values
    return list_values

def read_params(param_file):
    yaml = YAML()
    yaml.preserve_quotes = True
    with open('./config/travis_CI/config_ci_segmentation_local.yaml') as fp:
        data = yaml.load(fp)
        fp.close()
    return data
def write_params(param_file, data):
    yaml = YAML()
    with open('./config/travis_CI/config_ci_segmentation_local.yaml', 'w') as fp:
        yaml.dump(data, fp)
        fp.close()

def save_samp_ims(data_location, experiment_dir, set):

    dataset_ims_to_show = ('map_img', 'sat_img')

    f = h5py.File(data_location + '\\' + experiment_dir + '\\' + set + '_samples.hdf5', 'r')

    for dataset_name in dataset_ims_to_show:
        dataset = f[dataset_name]
        for imN in range(dataset.shape[0]):
            print(dataset_name)
            print(dataset[imN,...].shape[0])

    sample_html = open('hello.html', 'w+')
    sample_html.close()
    


if __name__ == '__main__':
# 0) read in params
    parser = argparse.ArgumentParser(description='Sample preparation')
    parser.add_argument('ParamFile', metavar='DIR',help='Path to training parameters stored in yaml')
    args = parser.parse_args()
    param_path = Path(args.ParamFile)
    print(args.ParamFile)
    params = read_params(args.ParamFile)

# 1) options
#-----------------------------------------------------------------------------------------------------------------------

    OPTS = {'HPC'               : False,
            'show model layers' : False,
            'vis_samples'       : True,
            'output_html'       : 'test',
            'out_html_dir'      : ''}


    # config =   {"l1": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
    #             "l2": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
    #             "lr": tune.loguniform(1e-4, 1e-1),
    #             "batch_size": tune.choice([2, 4, 8, 16])}

    path_name = 'samples'+ \
                str(params['global']['samples_size'])+ \
                '_overlap'+ \
                str(params['sample']['overlap'])+ \
                '_min-annot'+ \
                str(params['sample']['sampling_method']['min_annotated_percent'])+ \
                '_'+\
                str(params['global']['number_of_bands'])+ \
                'bands_'+ \
                str(params['global']['mlflow_experiment_name'])

# 2) set up console
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    console = Console(record=True) # width=...
    if OPTS['HPC']:  sys.stdout = open(os.devnull, "w")

# 3) run im-samp & prints
#-----------------------------------------------------------------------------------------------------------------------

    console.print('   DEBUG   =  ', params['global']['debug_mode'], style='bold purple', justify='left')

    console.print(' ',justify='center',style='on #FFFFFF')
    console.print(' ',justify='center',style='on #FFFFFF')
    console.print(' ',justify='left',style='bold #FFFFFF on #000000')
    console.print('STEP 1:',justify='center',style='bold #FFFFFF on #000000')
    console.print('MAKE SAMPLES',justify='center',style='bold #FFFFFF on #000000')
    console.print(' ',justify='left',style='bold #FFFFFF on #000000')
    console.print(' ',justify='center',style='on #FFFFFF')
    console.print(' ',justify='center',style='on #FFFFFF')

    # info Panel
    txt = Text('parent dir =\t')
    txt.append(params['global']['data_path'])
    txt.append('\nsmples dir =\t')
    txt.append(path_name)
    txt.append('\ncsv        =\t')
    txt.append(params['sample']['prep_csv_file'])
    params['sample']['prep_csv_file'] = 'C:/Users/muzwe/Documents/GitHub'
    if os.path.isdir(Path(params['global']['data_path']+'/'+path_name)):
        console.print(Panel(txt,title='NOT, prcessing new Samples', style='red'))
    else:
        console.print(Panel(txt,title='YES, prcessing new Samples', style='green'))
        IM_TO_SAMPLES_main(params, console)

    # output data panel
    trees = []
    num_samples = {}
    for sN, set in enumerate(['trn', 'tst', 'val']):
        trees.append(Tree(set, style='color('+str(sN+2)+')'))
        with h5py.File(params['global']['data_path'] + '/' + path_name + '/' + set + '_samples.hdf5', 'r') as f:
            for dataset_name in ('map_img', 'sat_img'):
                dataset = f[dataset_name]
                new = trees[sN].add(dataset_name)
                new.add('[white]'+str(dataset.shape[0]))
                new.add(str(dataset.shape))
                num_samples[set] = dataset.shape[0]

    console.print(Panel(Columns((trees[0], trees[1], trees[2]), equal=True, expand=True), title='Smples output'))

# 4) run training
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    console.print(' ',justify='center',style='on #FFFFFF')
    console.print(' ',justify='center',style='on #FFFFFF')
    console.print(' ',justify='left',style='bold #FFFFFF on #000000')
    console.print('STEP 2:',justify='center',style='bold #FFFFFF on #000000')
    console.print('TRAIN',justify='center',style='bold #FFFFFF on #000000')
    console.print(' ',justify='left',style='bold #FFFFFF on #000000')
    console.print(' ',justify='center',style='on #FFFFFF')
    console.print(' ',justify='center',style='on #FFFFFF')

    # assert model info
#     console.print(' ',justify='center',style='on #FFFFFF')
#     console.print('   model    =   ',params['global']['model_name'],justify='center',style='bold #FFFFFF on #000000')
#     console.print(' ',justify='center',style='on #FFFFFF')
#
#     if params['global']['model_name'] == 'deeplabv3_resnet101':
#         table = Table(title='[bold]deeplabv3_resnet101', show_lines=True, expand=True)
#         table.add_column('required stat',justify="center", style="cyan", no_wrap=True)
#         table.add_column('[#787878]op', justify="center", style="cyan", no_wrap=True)
#         table.add_column('requirement',justify="center", style="cyan", no_wrap=True)
#         table.add_column('assert',justify="center", style="cyan", no_wrap=True)
#
#         table.add_row('range', '=', str([0, 1]), str([0, 1]==params['global']['scale_data']))
#         table.add_row('mean', '=', str([0.485, 0.456, 0.406]))#, str(np.equal([0.485, 0.456, 0.406], params['global']['scale_data'])))
#         table.add_row('std', '=', str([0.229, 0.224, 0.225]))#, str(np.equal([0.229, 0.224, 0.225], params['global']['scale_data'])))
#         table.add_row('min_pixel_res', '>=', str(224))#,str(224>=0))
#
#         console.print(table)
#     console.print(' ',justify='center',style='on #FFFFFF')

    # info Panel
    list = Table(expand=True, show_lines=True, style="orchid1")
    list.add_column('[bold]catagory',justify='center', no_wrap=False)
    list.add_column('[bold]path',justify='center', no_wrap=False)
    list.add_column('[bold]exists?',justify='center', no_wrap=False)

    list.add_row('parent dir',
                 str(params['global']['data_path']),
                 str(Path(params['global']['data_path']).is_dir()))
    list.add_row('samples dir',
                 str(path_name),
                 str(Path(params['global']['data_path']).joinpath(path_name).is_dir()))
    list.add_row('model dir',
                 'model_' + str(params['global']['model_output_dir']),
                 str(Path(params['global']['data_path']).joinpath(path_name).joinpath('model_' + str(params['global']['model_output_dir'])).is_dir()))

    txt = Text(justify='center')
    # txt.append_text(Text('model  = ' + str(params['global']['model_name']) + '\n', style='bold cyan2'))
    # txt.append_text(Text('loss   = ' + str(params['training']['loss_fn']) + '\n', style='bold cyan2'))
    # txt.append_text(Text('optmzr = ' + str(params['training']['optimizer']) + '\n', style='bold cyan2'))

    def list_layers(model, count):
        modules = dict(model.named_modules())
        keys = []
        for key in modules:
            if key == '':  continue
            console.print(key, justify='right')#, style='color('+str(count)+')')
            console.print(modules[key], justify='left')#, style='color('+str(count)+')')
            # console.print(dict(modules[key].named_modules()), justify='center', style='color('+str(count)+')')
            console.print(count, justify='center', style='on color('+str(count)+')')
            list_layers(modules[key], count+1)
            console.print(count, justify='center', style='on color('+str(count)+')')

    # model layers
    if OPTS['show model layers']:
        txt.append('\n\n')

        model, model_name, criterion, optimizer, lr_scheduler = net(params, params['global']['num_classes']+1)
        # console.print(len(dict(model.named_modules())))
        # console.print(dict(model.named_modules()))
        # console.print()
        list_layers(model, 0)
        sys.exit()
        try:
            summary = torch_summary(model, (params['global']['number_of_bands'], params['global']['samples_size'], params['global']['samples_size']))
            table = Table(title=params['global']['model_name'], expand=True)
            table.add_column("Layer (type)", justify="center", style="bright_cyan", no_wrap=True)
            table.add_column("Output Shape", justify="center", style="bright_cyan", no_wrap=True)
            table.add_column("Param #", justify="center", style="bright_cyan", no_wrap=True)

            for layer in summary:
                table.add_row(layer, str(summary[layer]["output_shape"]), "{0:,}".format(summary[layer]["nb_params"]))

            console.print(Panel(RenderGroup(txt,table, summary['final_summary']),
                                title='pre-training info', box=box.DOUBLE_EDGE, style="magenta1"))
            console.print(str(model))
        except AttributeError:
            console.print(Panel(RenderGroup(txt,Text('model = ' + str(params['global']['model_name']))),
                                title='pre-training info', box=box.DOUBLE_EDGE, style="magenta1"))
            console.print(str(model))
    else:
        console.print(Panel(RenderGroup(list, txt),
                            title='pre-training info', box=box.DOUBLE_EDGE, style="magenta1"))

    changes = {}
    changes['learning_rate'] = [0.0001]
    changes['weight_decay'] = [0]
    changes['step_size'] = [4]
    changes['gamma'] = [0.9]

    experiments = Table('exp. num.', 'model', 'optimzier', 'loss func',
                        'learning_rate', 'weight_decay', 'step_size', 'gamma',
                        title='experiments', expand=True, style='purple')

    experiments.add_row(str(1),
                        str(params['training']['learning_rate']),
                        str(params['training']['weight_decay']),
                        str(params['training']['step_size']),
                        str(params['training']['gamma']))

    console.print(experiments)

    for change in changes:
        params['training'][change] = changes[change][0]

    # if params['training']['loss_fn'] == 'Lovasz':
    #     params['training']['class_weights'] = None



    trckr = h5py.File('output_path', 'w')
    # for set in ['trn', 'tst', 'val']:
    #     trckr.create_group(set)
    trckr.create_dataset('acc')
    trckr.create_dataset('pers')
    trckr.create_dataset('iou')
    trckr.create_dataset('fscore')

    write_params(args.ParamFile, params)
    TRAIN_main(params, param_path, console, trckr) # TODO: make sure model_NAME doesnt exist already



#-----------------------------------------------------------------------------------------------------------------------

    # console.export_html(clear=False)
    # console.save_html(OPTS['output_html']+'.html', clear=False)


# #-----------------------------------------------------------------------------------------------------------------------
#
#     from rich.tree import Tree
#     tree = Tree("Rich Tree")
#     baz_tree = tree.add("baz")
#     tree.add("baz")
#     baz_tree.add("[red]Red").add("[green]Green").add("[blue]Blue")
#     # console.print(tree)
#
# #-----------------------------------------------------------------------------------------------------------------------
#
#     from rich.columns import Columns
#     columns = Columns((tree, tree), equal=True, expand=True)
#     # console.print(columns)
#
# #-----------------------------------------------------------------------------------------------------------------------
#
#     from rich.panel import Panel
#     panel = Panel(columns, title='[bold]just an example')
#
#     console.print(panel)
#
# #-----------------------------------------------------------------------------------------------------------------------
#
#     from rich.table import Table
#     table = Table(title="[bold]\nThe Worst Star Wars[/bold] Movies", show_lines=True)
#     table.add_column("Released", justify="center", style="cyan", no_wrap=True)
#
#     table.add_row("Dec 20, 2019", "Star Wars: The Rise of Skywalker")
#     table.add_row("May 25, 2018", "Solo: A Star Wars Story", "$393,151,347")
#     console.print(table)