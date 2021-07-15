## **How to use *Hydra* config files**
When using [*Hydra*](https://hydra.cc/docs/intro/#quick-start-guide), we have the main `yaml` file [gdl_config_template.yaml](gdl_config_template.yaml) that will handle the other `yaml` files in the dedicated folders.
We recommend using the template as the main config and run your code with the proper parameters, see the main [README](../README.md) to see how.
Otherwise, we will show some example at the end of the document ([Examples](#Examples)).

The structure of the template is as follows:
```YAML
defaults:
      ...
  
general:
      ...

inference:
      ...
    
task:  ...
mode:  ...
debug: ...
```

## Defaults
```YAML
defaults:
  -hydra: default
  -model: fastrcnn
  -trainer: default_trainer
  -training: default_training
  -optimizer: adamw
  -callbacks: default_callbacks
  -scheduler: plateau
  -sampling: default_sampling
  -data: data
  -dataset: NRCAN_individual_tree
  -augmentation: default_augmentation
  -logging: mlflow
```
The defaults section will automatically load the `yaml` files specified in the [gdl_config_template.yaml](gdl_config_template.yaml). 
Those files need to be inside folders that are in the [`config`](../config) folder.
```
├── config
    └── gdl_config_template.yaml
    └── hydra
        └── default.yaml
        └── save.yaml
    └── model
        └── unet.yaml
        └── fastrcnn.yaml
      ...
```
Those files can be somewhere else but need to be specified in argument like show in the [Examples](#Examples) section.
So if you want to add some configuration that can be useful to **GDL**, we recommend adding a default `yaml` file with default parameters.
Keep in mind that you need to include `# @package _global_` at the beginning of each `yaml` added. 
Like that *Hydra* will interpret the `yaml` file like the option where you have assigned the new file.
For example if you created `new_model.yaml` to be read as a model, and you don't want the change the main code to read this file each time you change model.
With just `# @package _global_` the python code will read `cfg.model`.
For more information on how to write a `yaml` file for each default parameters, a `README.md` will be in each folder.

## General
```YAML
general:
  project_name: template_project
  workspace: your_name
  device: cuda
  max_epochs: 10
  min_epochs: 1
  save_dir: './log'
  sample_data_dir: 'img/dir'
  save_weights_dir: './weights_saved'
```
This section regroups general information that will be read by the code, other `yaml` files read information from here.

## Inference
```YAML
inference:
  image_path: 'image/path'
  save_dir:
  weights_dir: ${general.save_weights_dir}
  weights_name:
```
The inference section have the information to execute the inference job (more option will follow soon).
The `image_path` is the path where are the image(s).
The parameters `save_dir` is where you want the inference image(s), if not specify, the image(s) will be saved where the code is run.
For the `weights_dir`, the code will read the same folder that the folder in `general`, but if you only want to do the inference you can change this path only.
If `weights_name` is not specify, it will load the last weight saved in the `weights_dir`.

## Task
```YAML
task: {segmentation, object_detection, point_cloud, super_resolution}
```
For now **GDL** can do segmentation, object detection, point cloud and super resolution task.
To launch the main code, you need to specify the task, and the task need to be in this list. 

## Mode
```YAML
mode: {sampling, train, evaluate, inference, hyperparameters_search}
```
**GDL** have five mode: sampling, train, evaluate, inference and hyperparameters search.
- *sampling*, will generate `hdf5` files from a folder containing folders for each individual image with their ground truth.
- *train*, will train the model specify with all the parameters in `training`, `trainer`, `optimizer`, `callbacks` and `scheduler`. The outcome will be `.pth` weights.
- *evaluate*, this function need to be fill with images, their ground truth and a weight for the model. At the end of the evaluation you will obtain statistics on those images. 
- *inference*, unlike the evaluation, the inference don't need a ground truth. The inference will produce a prediction on the content of the images feed to the model. Depending on the task, the outcome file will differ.
- *hyperparameters search*, soon to come.

>Each of those modes will be different for all the tasks, a documentation on how to write a new task with all the proper functions will follow soon.

## Debug
```YAML
debug: False
```
Will print the complete yaml config plus run a validation test before the training.

##Examples
Here some examples on how to run **GDL** with *Hydra*.

- Basic, run the code with all the defaults value in the [`gdl_config_template.yaml`](gdl_config_template.yaml).
```bash
$ python GDL.py mode=train
```
- Changing only one parameter.
```bash
$ python GDL.py mode=train general.max_epochs=100
```
- Want to add a new parameters in the config without having to writing in the `yaml`. 
You can also use a path to a `yaml` instead of `1.0` that will create a group like general or inference.
```bash
$ python GDL.py mode=train +new.params=1.0
```
```YAML
general:
      ...

inference:
      ...
    
task:  ...
mode:  ...
debug: ...
new:
  params: 1.0
```

- New `gdl_config.yaml` file that have the structure that the template.  
```bash
$ python GDL.py cfgs=/path/to/new/gdl_config.yaml mode=train
```
