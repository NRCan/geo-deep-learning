## **How to use *Hydra* config files**
How [*Hydra*](https://hydra.cc/docs/intro/#quick-start-guide) is working is that we have the main `yaml` file ([gdl_config_template.yaml](gdl_config_template.yaml)) that will handle the other `yaml` files in the dedicated folders.
We recommend using the template as the main config and run your code with the proper parameters, see the main [README](../README.md) to see how to use it for a segmentation task.
Otherwise, we will show some example at the end of the document in the [Examples](#Examples) section.

First, the config folder is structured like the following example. 
Keeping in mind inside the config folder you will have the `gdl_config.yaml` that will regroup every parameter for executing the command.
Another think you will find are folder containing `yaml` files for certain categories of parameters, this will be explain in the next section [Defaults Section](#Defaults Section).
```
├── config
    └── gdl_config_template.yaml
    └── model
        └── unet.yaml
        └── fastrcnn.yaml
      ...
    └── task
        └── segmentation.yaml
          ...
```

The code is base to read [gdl_config_template.yaml](gdl_config_template.yaml) by default when executed, if you want to create your own `gdl_config.yaml` see the [Examples](#Examples) section.
But keep in mind when creating your own config to keep the following structure.
```YAML
defaults:
      ...
general:
      ...
inference:
      ...
AWS: 
      ...
print_config: ...
mode:  ...
debug: ...
```

#### Defaults Section
The **_'defaults'_** part is the part where all the default `yaml` file are loaded as reading values.
Example, `task: segmentation` mean that inside the `config` folder you have another folder name `task` and inside it, we have the `segmentation.yaml` file.
This is the same for all the bullets points in the `defaults` section.
Inside the bullet folder, you will see all the option for that categories of parameters, like the `model` categories will regroup all the `yaml` file with the parameters for each model.
```YAML
defaults:
  - task: segmentation
  - model: unet
  - trainer: default_trainer
  - training: default_training
  - optimizer: adamw
  - callbacks: default_callbacks
  - scheduler: plateau
  - dataset: test_ci_segmentation_dataset
  - augmentation: basic_augmentation_segmentation
  - tracker: # set logger here or use command line (e.g. `python run.py tracker=mlflow`)
  - visualization: default_visualization
  - inference: default_inference
  - hydra: default
  - override hydra/hydra_logging: colorlog # enable color logging to make it pretty
  - override hydra/job_logging: colorlog # enable color logging to make it pretty
  - _self_
```
Those files can be somewhere else but need to be specified in argument like show in the [Examples](#Examples) section, and the main goal of this new structure is to organise all the parameters in categories to be easier to find.
If you want to add new option for a categories, keep in mind that you need to include `# @package _global_` at the beginning of each `yaml` added. 
Like that, the code in python will read `model.parameters_name` like a directory, otherwise without the `# @package _global_` the python code will read `model.unet.parameters_name`.
This will be going the opposite to what we want, the objective is to not associate a specific name to a model in this example but keeping it general.
For example if you created `new_model.yaml` to be read as a model, and you don't want the change the main code to read this file each time you change model.
For more information on how to write a `yaml` file for each default parameters, a `README.md` will be in each categories folder.

The **_tracker section_** is set to `None` by default, but still will log the information in the log folder.
If you want to set a tracker you can change the value in the config file or to add the tracker parameter in the command line `python GDL.py tracker=mlflow mode=train`.

The **_inference section_** have the information to execute the inference job (more option will follow soon).
This part doesn't need to be fill if you want to only launch sampling, train or hyperparameters search mode.

The **_task section_** manage the executing task, by default we put `segmentation` since it's the primary task of GDL.
But the objective will be to evolve during time and task will be added, like that it will be easy to manage.
The `GDL.py` code will work like executing the main function from the `task_mode.py` in the main folder of GDL.
The chosen `yaml` from the task categories, will regroup all the parameters only proper to the wanted task.

#### General Section
```YAML
general:
  work_dir: ${hydra:runtime.cwd}
  config_name: ${hydra:job.config_name}
  config_override_dirname: ${hydra:job.override_dirname}
  config_path: ${hydra:runtime.config_sources}
  project_name: template_project
  workspace: your_name
  device: cuda
  max_epochs: 2 # for train only
  min_epochs: 1 # for train only
  raw_data_dir: ${general.work_dir}/data
  raw_data_csv: ${general.work_dir}/data/images_to_samples_ci_csv.csv
  sample_data_dir: ${general.work_dir}/data
  state_dict_path:
  save_weights_dir: ${general.work_dir}/weights_saved
```
This section regroups general information that will be read by the code, other `yaml` files read information from here.

#### AWS Section
Will follow soon.

#### Print Config Section
If `True`, will save the config in the log folder.

#### Mode Section
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

#### Debug Section
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
defaults:
      ...

general:
      ...
    
print_config:  ...
mode:  ...
debug: ...
new:
  params: 1.0
```

- New `gdl_config.yaml` file that have the structure that the template.  
```bash
$ python GDL.py cfgs=/path/to/new/gdl_config.yaml mode=train
```


