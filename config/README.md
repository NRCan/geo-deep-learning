## **How to use *Hydra* config files**
GDL configures [*Hydra*](https://hydra.cc/docs/intro/#quick-start-guide) through a main `yaml` file ([gdl_config_template.yaml](gdl_config_template.yaml)) located in the `config` folder which handles additional `yaml` files located in other folders.
We recommend starting with the [main yaml file](gdl_config_template.yaml) as the configuration entrypoint. Run your code with the proper parameters - see the main [README](../README.md) for an example with a semantic segmentation task.
There are other examples at the end of this document in the [Examples](#Examples) section.

The `config` folder is structured as depicted below. It is important to remember that the `gdl_config_template.yaml` file contains every parameter necessary for executing the command.  Other `yaml` files in subdirectories handle specific categories of parameters. See details in the next section [Defaults Section](#Defaults Section).
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

The code is currently executed with [gdl_config_template.yaml](gdl_config_template.yaml) by default. If you want to create your own `gdl_config.yaml` see the [Examples](#Examples) section.
But keep in mind your own config will require the following structure:
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
The **_'defaults'_** section is where all default `yaml` files are loaded as input values for each category of parameters.
The same is true for all items in the `defaults` section.
For example, `task: segmentation` means the `config` folder contains a subfolder called `task` which contains a `segmentation.yaml` file.
Options for each category of parameters are found in config subfolders.  The `model` category bundles all the `yaml` files with the parameters for each model.
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
The files in the `defaults` section can be overwritten on the command line. See [Examples](#Examples) section. The main goal of the structure is to organize all the parameters in meaningful categories.
If you want to add new options for a category, you'll need to include `# @package _global_` at the beginning of each `yaml` added. 
By doing so, the code in python will read `model.parameters_name` as a directory. If you accidentally omit the prefix `# @package _global_`, the python code will read `model.unet.parameters_name` (as set by default currently).
For example if you created `new_model.yaml` to be read as a model and you don't want to change the main code to read this file each time you change model.
For more information on how to write a `yaml` file for each default parameters, a `README.md` will be in each category folder.
For more information about packages in Hydra, see [Hydra's documentation on Packages](https://hydra.cc/docs/advanced/overriding_packages)

The **_tracker section_** is set to `None` by default, but will still log the information in the log folder.
If you want to set a tracker you can change the value in the config file or add the tracker parameter at execution time via the command line `python GDL.py tracker=mlflow mode=train`.

The **_inference section_** contains the information to execute the inference job (more options will follow soon).
This part doesn't need to be filled if you want to launch tiling, train or hyperparameters search mode only.

The **_task section_** manages the executing task. `Segmentation` is the default task since it's the primary task of GDL.
However, the goal will be to add tasks as need be. The `GDL.py` code simply executes the main function from the `task_mode.py` in the main folder of GDL.
The chosen `yaml` from the task categories will gather all the parameters relevant (as chosen by user) for the desired task.

#### General Section
```YAML
general:
  work_dir: ${hydra:runtime.cwd}  # where the code is executed
  config_name: ${hydra:job.config_name}
  config_override_dirname: ${hydra:job.override_dirname}
  config_path: ${hydra:runtime.config_sources}
  project_name: template_project
  workspace: your_name
  max_epochs: 2 # for train only
  min_epochs: 1 # for train only
  raw_data_dir: data
  raw_data_csv: tests/sampling/sampling_segmentation_binary_ci.csv
  sample_data_dir: data # where the hdf5 will be saved
  state_dict_path:
  save_weights_dir: saved_model/${general.project_name}
```
This section contains general information that will be read by the code. Other `yaml` files read information from here.

#### AWS Section
Will follow soon.

#### Print Config Section
If `True`, will save the config in the log folder.

#### Mode Section
```YAML
mode: {tiling, train, inference, evaluate, hyperparameters_search}
```
**GDL** has five modes: tiling, train, evaluate, inference and hyperparameters search.
- *tiling*, generates .geotiff and .geojson tiles from each individual aoi (image & ground truth).
- *train*, will train the model specified with all the parameters in `training`, `trainer`, `optimizer`, `callbacks` and `scheduler`. The outcome will be `.pth` weights.
- *evaluate*, this function needs to be filled with images, their ground truth and a weight for the model. At the end of the evaluation you will obtain statistics on those images. 
- *inference*, unlike the evaluation, the inference doesn't need a ground truth. The inference will produce a prediction on the content of the images fed to the model. Depending on the task, the outcome file will differ.
- *hyperparameters search*, soon to come.

>Each of those modes will be different for all the tasks, a documentation on how to write a new task with all the proper functions will follow soon.

#### Debug Section
```YAML
debug: False
```
Will print the complete yaml config plus run a validation test before the training.

## Examples
Here some examples on how to run **GDL** with *Hydra*.

- For basic usage, run the code with all the defaults value in the [`gdl_config_template.yaml`](gdl_config_template.yaml).
```bash
$ python GDL.py mode=train
```
- Overriding only one parameter.
```bash
$ python GDL.py mode=train general.max_epochs=100
```
- Adding a new parameters in the config without having to write it in the `yaml`. 
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

- Using a new `gdl_config.yaml` file that has the same structure as the template. For more information, see 
- [Hydra's documentation on command line flags](https://hydra.cc/docs/advanced/hydra-command-line-flags/)  
```bash
$ python GDL.py --config-name=/path/to/new/gdl_config.yaml mode=train
```


