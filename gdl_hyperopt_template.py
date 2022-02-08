"""Hyperparamater optimization for GDL using hyperopt

This is a template for using hyperopt with GDL. The my_space variable currently needs to
be modified here, as well as GDL config modification logic within the objective_with_args
function.

"""

import argparse
from pathlib import Path
import pickle
from functools import partial
import pprint
import numpy as np

import mlflow
import torch
# ToDo: Add hyperopt to GDL requirements
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from ruamel_yaml import YAML

from train_segmentation import main as train_main

# This is the hyperparameter space to explore
my_space = {'model_name': hp.choice('model_name', ['unet_pretrained', 'deeplabv3_resnet101']),
            'loss_fn': hp.choice('loss_fn', ['CrossEntropy', 'Lovasz', 'Duo']),
            'optimizer': hp.choice('optimizer', ['adam', 'adabound']),
            'learning_rate': hp.loguniform('learning_rate', np.log(1e-7), np.log(0.1))}


def get_latest_mlrun(params):
    """Get latest mlflow run

    :param params: gdl parameters dictionary
    :return: mlflow run object
    """

    tracking_uri = params['global']['mlflow_uri']
    mlflow.set_tracking_uri(tracking_uri)
    mlexp = mlflow.get_experiment_by_name(params['global']['mlflow_experiment_name'])
    exp_id = mlexp.experiment_id
    try:
        run_ids = ([x.run_id for x in mlflow.list_run_infos(
            exp_id, max_results=1, order_by=["tag.release DESC"])])
    except AttributeError:
        mlflow_client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        run_ids = [x.run_id for x in mlflow_client.list_run_infos(exp_id, run_view_type=3)[0:1]]
    mlrun = mlflow.get_run(run_ids[0])
    return mlrun


def objective_with_args(hparams, params, config_path):
    """Objective function for hyperopt

    This function edits the GDL parameters and runs a training.

    :param hparams: arguments provided by hyperopt selection from hyperparameter space
    :param params: gdl parameters dictionary
    :param config_path: path to gdl configuration file
    :return: loss dictionary for hyperopt
    """

    # ToDo: This is dependent on the specific structure of the GDL config file
    params['global']['model_name'] = hparams['model_name']
    # params['training']['target_size'] = hparams['target_size']
    params['training']['loss_fn '] = hparams['loss_fn']
    params['training']['optimizer'] = hparams['optimizer']
    params['training']['learning_rate'] = hparams['learning_rate']

    try:
        mlrun = get_latest_mlrun(params)
        run_name_split = mlrun.data.tags['mlflow.runName'].split('_')
        params['global']['mlflow_run_name'] = run_name_split[0] + f'_{int(run_name_split[1]) + 1}'
    except:
        pass
    train_main(params, config_path)
    torch.cuda.empty_cache()

    mlflow.end_run()
    mlrun = get_latest_mlrun(params)

    # ToDo: Probably need some cleanup to avoid accumulating results on disk

    # ToDo: This loss should be configurable
    return {'loss': -mlrun.data.metrics['tst_iou'], 'status': STATUS_OK}


def trials_to_csv(trials, csv_pth):
    """hyperopt trials to CSV

    :param trials: hyperopt trials object
    """

    params = sorted(list(trials.vals.keys()))
    csv_str = ''
    for param in params:
        csv_str += f'{param}, '
    csv_str = csv_str + 'loss' + '\n'

    for i in range(len(trials.trials)):
        for param in params:
            if my_space[param].name == 'switch':
                csv_str += f'{my_space[param].pos_args[trials.vals[param][i] + 1].obj}, '
            else:
                csv_str += f'{trials.vals[param][i]}, '
        csv_str = csv_str + f'{trials.results[i]["loss"]}' + '\n'

    # ToDo: Customize where the csv output is
    with open(csv_pth, 'w') as csv_obj:
        csv_obj.write(csv_str)


def main(params, config_path):
    # ToDo: Customize where the trials file is
    # ToDo: Customize where the trials file is
    root_path = Path(params['global']['assets_path'])
    pkl_file = root_path.joinpath('hyperopt_trials.pkl')
    csv_file = root_path.joinpath('hyperopt_results.csv')
    if pkl_file.is_file():
        trials = pickle.load(open(pkl_file, "rb"))
    else:
        trials = Trials()

    objective = partial(objective_with_args, params=params, config_path=config_path)

    n = 0
    while n < params['global']['hyperopt_runs']:
        best = fmin(objective,
                    space=my_space,
                    algo=tpe.suggest,
                    trials=trials,
                    max_evals=n + params['global']['hyperopt_delta'])
        n += params['global']['hyperopt_delta']
        pickle.dump(trials, open(pkl_file, "wb"))

    # ToDo: Cleanup the output
    pprint.pprint(trials.vals)
    pprint.pprint(trials.results)
    for key, val in best.items():
        if my_space[key].name == 'switch':
            best[key] = my_space[key].pos_args[val + 1].obj
    pprint.pprint(best)
    print(trials.best_trial['result'])
    trials_to_csv(trials, csv_file)


def read_parameters(param_file):
    """Read and return parameters in .yaml file
    Args:
        param_file: Full file path of the parameters file
    Returns:
        YAML (Ruamel) CommentedMap dict-like object
    """
    yaml = YAML()
    with open(param_file) as yamlfile:
        params = yaml.load(yamlfile)
    return params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Geo Deep Learning hyperopt')
    parser.add_argument('param_file', type=str, help='Path of gdl config file')
    args = parser.parse_args()
    gdl_params = read_parameters(args.param_file)
    gdl_params['self'] = {'config_file': args.param_file}
    main(gdl_params, Path(args.param_file))
