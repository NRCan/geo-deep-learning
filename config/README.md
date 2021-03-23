## **Preparation of the `csv` file**
The `csv` file specifies the input images and the reference vectors data that will be use during the training, validation and testing phases.
Each row in the `csv` file must contain 5 comma-separated items for the five items that follow.
- path to the input image file (tif)
- (*Optional*) metadata info, leave empty if not desired. (see example below)
- path to the vectors data (GeoPackage)
- attribute of the GeoPackage to use as classes values
- where to use the dataset : '*trn*' for training, '*val*' for validation or '*tst*' for test  

Each image is a new line like:

```
\path\to\input\image1.tif,,\path\to\reference\vector1.gpkg,attribute,trn
\path\to\input\image2.tif,,\path\to\reference\vector2.gpkg,attribute,val
\path\to\input\image3.tif,,\path\to\reference\vector2.gpkg,attribute,tst
```
> **Note :** '*tst*' is optional for training the neural network, but if you want only use the inference, the code will only use the images from '*tst*'.

---

## **Preparation of the `yaml` file**
The `config_template.yaml` file located in the `config` directory. We highly recommend to copy that file and past it inside a folder with all the information of the training (see the `Folder Structure` in the [README.md](../README.md#Folder-Structure) the main directory).

So this `yaml` file stores the values of all parameters needed by the deep learning algorithms for the three phases. The file contains the following sections:
* [Global](#Global)
* [Data Analysis](#Data-Analysis)
* [Sampling](#Sampling)
* [Training](#Training)
* [Inference](#Inference)
* [Visualization](#Visualisation)

<!-- Specific parameters in each section are shown below, where relevant. For more information about config.yaml, view file directly: [config/config.yaml](https://github.com/NRCan/geo-deep-learning/blob/master/config/config.yaml) -->

## **Global**
This section regroup all the general information and use by more that one sections.

```YAML
global:
  samples_size: 512
  num_classes: 5
  data_path: path/to/data
  number_of_bands: 4
  model_name: unet
  mlflow_uri: path/to/mlflow_tracking_uri
  mlflow_experiment_name: gdl-training
  mlflow_run_name: gdl
  bucket_name:
  task: segmentation
  num_gpus: 2
  max_used_ram: 2000
  max_used_perc: 15
  BGR_to_RGB: True
  scale_data: [0,1]
  aux_vector_file:
  aux_vector_attrib:
  aux_vector_ids:
  aux_vector_dist_maps:
  aux_vector_dist_log:
  aux_vector_scale:
  debug_mode: True
  coordconv_convert: False
  coordvonc_scale:
```
- **`samples_size` :** Size of the tiles that you want to train/val/test with. We recommend `512` or `256` for the train/val task.

- **`num_classes` :** Number of classes that your *ground truth* have and which you want to predict at the end.

- **`data_path` :** Path to the folder where samples folder will be automatically created and all the other informations will be stored. **add the link to the example of stucture**

- **`number_of_bands` :** Number of bands in input images (RGB = 3 and RGBNir = 4).

- **`model_name` :** Name of the model use to train the neural network, see the list of all the implemented models in [`/models`](../models#Models-available).

- **`mlflow_uri` :** Path where *mlflow* will store all the informations about the runs. By default the path is `./mlruns`.

- **`mlflow_experiment_name` :** Experiment name in *mlflow*.

- **`mlflow_run_name` :** Run name in *mlflow* and preprocessing identification tag.

- **`bucket_name` (Optional) :** Name of the S3 bucket where the data is stored. Leave blank if using local files.

- **`task` :** Task to perform, either segmentation or classification, but classification is no longer supported.

- **`num_gpus` :** Number of **GPUs** that you want to use, `0` will use the **CPU**.

- **`max_used_ram` :** Maximum currently used **GPUs** memory (MB) to consider it available.

- **`max_used_perc` :** Maximum utilization rate (percent) of the **GPUs** to consider it available.

- **`BGR_to_RGB` :**  [True/False] If set to True this parameter changes the band order of images from BGR to RGB.

- **`scale_data` :** Min and Max for input data rescaling, by default: `[0, 1]` meaning no rescaling.

- **`aux_vector_file` (Optional) :** A vector file from which to extract auxiliary shapes.

- **`aux_vector_attrib` (Optional) :** A vector file attribute name to parse in order to fetch ids.

- **`aux_vector_ids` (Optional) :** A vector ids to target in the vector file above.

- **`aux_vector_dist_maps` (Optional) :** [True/False] Flag indicating whether aux vector bands should be distance maps or binary maps.

- **`aux_vector_dist_log` (Optional) :** [True/False] Flag indicating whether log distances should be used in distance maps or not.

- **`aux_vector_scale` (Optional) :** Floating point scale factor to multiply to rasterized vector maps.

- **`debug_mode` :** Activates various debug features for example, details about intermediate outputs, detailed progress bars, etc. By default this mode is `False`.

- **`coordconv_convert` (Optional):** [True/False] Activate (or not) the function [`swap_coordconv_layers`](../models/coordconv.py#L94).

- **`coordvonc_scale` (Optional) :** Scale of the map.

## **Data Analysis**
The [data_analysis](data_analysis.py) module is used to visualize the composition of the sample's classes and see how it shapes the training dataset and can be  useful for balancing training data in which a class is under-represented. Using basic statistical analysis, the user can test multiple sampling parameters and immediately see their impact on the classes' distribution. It can also be used to automatically search optimal sampling parameters and obtain a more balanced class distribution in the dataset.

The sampling parameters can then be used in [images_to_samples.py](../images_to_samples.py) to obtain the desired dataset or can be use alone this way there is no need to run the full code to find out how the classes are distributed (see the [example](#Running)).

Before running [data_analysis.py](data_analysis.py), the paths to the `csv` file containing all the information about the images and the data folder must be specified in the `yaml` file use to the experience.
```YAML
# Global parameters
global:
  samples_size: 512
  num_classes: 5
  data_path: path/to/data                 # <--- must be specified
  number_of_bands: 4

      ...

# Sample parameters; used in images_to_samples.py -------------------
sample:
  prep_csv_file: /path/to/csv/images.csv  # <--- must be specified
  val_percent: 5

```

Here is an example of the data_analysis section in the `YAML` file :

```YAML
data_analysis:
  create_csv: True
  optimal_parameters_search : False
  sampling_method:
    'min_annotated_percent': 0
    'class_proportion': {'1':0, '2':0, '3':0, '4':0}
```
- **`create_csv` :**
This parameter is used to create a `csv` file containing the class proportion data of each image sample. This first step is mandatory to ensure the proper operation of the module. Once it is created, the same `csv` file is used for every tests the user wants to perform. Once that is done, the parameter can then be changed to `False`.
This parameter would have to be changed to `True` again if any changes were made to the content of the `prep_csv_file` or if the user wishes to change the values of the `samples_size` parameters. These parameters have a direct effect on the class proportion calculation. The `csv` file created is stored in the folder specified in the `data_path` of the global section.

- **`optimal_parameters_search` :**
When this parameter is set to `True`, it activates the optimal sampling parameters search function. This function aims to find which sampling parameters produce the most balanced dataset based on the standard deviation between the proportions of each class in the dataset. The sampling method(s) used for the search function must first be specified in the `sampling_method` dictionary in the `data_analysis` section. It does not take into account the values of the other keys in the dictionary. The function first returns the optimal threshold(s) for the chosen sampling method(s). It then returns a preview of the proportions of each classes and the size of the final dataset without creating it, like the following image.
<p align="center">
   <img align="center" src="/docs/screenshots/stats_parameters_search_map_cp.PNG">
</p>

- **`sampling_method` :**
For `min_annotated_percent` is the minimum percent of non background pixels in the samples. By default the value is `0` and the targeted minimum annotated percent must by a integer.
`class_proportion` should be a dictionary with the number of each classes in the images in quotes. Specify the minimum class proportion of one or all classes with integer(s) or float(s). Example, `'0':0, '1':10, '2':5, '3':5, ...`
<!-- `min_annotated_percent`, For this value to be taken into account, the `optimal_paramters_search` function must be turned off and 'min_annotated_percent' must be listed in the `'method'` key of the `sampling` dictionary. -->
<!-- `class_proportion`, For these values to be taken into account, the `optimal_paramters_search` function must be turned off and 'class_proportion' must be listed in the `'method'` key of the `sampling` dictionary. -->

#### Running [data_analysis.py](data_analysis.py)
You can run the data analysis alone if you only want the stats, and to do that you only need to launch the program :
```shell
python utils/data_analysis.py path/to/yaml_files/your_config.yaml
```


## **Sampling**
This section is use by [images_to_samples.py](../images_to_samples.py) to prepare the images for the training, validation and inference. Those images must be geotiff combine with a GeoPackage, otherwise it will not work.

In addition, [images_to_samples.py](../images_to_samples.py) will assert that all geometries for features in GeoPackages are valid according to [Rasterio's algorithm](https://github.com/mapbox/rasterio/blob/d4e13f4ba43d0f686b6f4eaa796562a8a4c7e1ee/rasterio/features.py#L461).

```yaml
sample:
  prep_csv_file: path/to/images.csv
  val_percent: 5
  overlap: 25
  sampling_method:
    'min_annotated_percent': 0
    'class_proportion': {'1':0, '2':0, '3':0, '4':0}
  mask_reference: False
```
- **`prep_csv_file` :** Path to your `csv` file with the information on the images.

- **`val_percent` :** Percentage of validation samples created from train set (0 - 100), we recommend at least `5`, must be an integer (int).

- **`overlap` :** Percentage of overlap between 2 chunks, must be an integer (int).

- **`sampling_method` :** Chose one of the following method.
  - `min_annotated_percent` is the percentage minimum of non background pixels in samples by default we chose `0`, must be an integer (int).

  - `class_proportion` is a dictionary (dict) where the keys (numerical values in 'string' format) represent class id and the values (int) represent the class minimum threshold targeted in samples. An example of four classes with no minimum: `{'1':0, '2':0, '3':0, '4':0}`.

- **`mask_reference` :** A mask that mask the input image where there is no reference data, when the value is `True`.

#### Running [images_to_samples.py](../images_to_samples.py)
You must run this code before training to generate the `hdf5` use by the other code.
Even if you only want to do inference on the images, you need to generate a `tst_samples.hdf5`.

To launch the code:
```shell
python images_to_samples.py path/to/yaml_files/your_config.yaml
```
The output of this code will result at the following structure:
```
├── {data_path}
    └── {samples_folder}
        └── trn_samples.hdf5
        └── val_samples.hdf5
        └── tst_samples.hdf5
```
The name of the `{samples_folder}` will be written by the informations chosen like :
`samples{samples_size}_overlap{overlap}_min-annot{min_annot_perc}_{num_bands}bands`

>If folder already exists, a suffix with `_YYYY-MM-DD_HH-MM` will be added


## **Training**
This section is use by [train_segmentation.py](../train_segmentation.py) to train the chosen model. The training phase is the crux of the learning process.
During that phase, we can count two main actions:
- The main training, where the samples labeled "trn" are used to train the neural network.

- The validation, where the samples labeled "val" are used to estimate the training error (i.e. loss) on a set of sub-images (those image must not be used in the main training), after every epoch or after the number of epochs chosen.

At the end of each validation, the model with the lowest error on validation data
is save, if the network don't perform better than the last validation, the weights of the model will not be saved.

```yaml
training:
  state_dict_path: /path/to/checkpoint.pth.tar
  pretrained: True
  num_trn_samples: 4960
  num_val_samples: 2208
  num_tst_samples: 1000
  batch_size: 32   
  num_epochs: 150  
  target_size: 128
  loss_fn: Lovasz
  optimizer: adabound
  learning_rate: 0.0001
  weight_decay: 0     
  step_size: 4        
  gamma: 0.9
  dropout: False
  dropout_prob: False
  class_weights: [1.0, 0.9]
  batch_metrics: 2
  ignore_index: 0
  normalization:
    mean:
    std:
  augmentation:
    rotate_limit: 45
    rotate_prob: 0.5
    hflip_prob: 0.5
    random_radiom_trim_range: [0.1, 2.0]
    brightness_contrast_range:
    noise:
```
- **`state_dict_path` (Optional):** Path to checkpoint (weights) from a trained model as .pth.tar or .pth file.

- **`pretrained` :** When `True` and the chosen model have the option available,  the model will load pretrained weights (e.g. Deeplabv3 pretrained on coco dataset). The default value is `True` if no `state_dict_path` is given.

- **`num_trn_samples` :** Number of samples to use for training by default all samples in `hdf5` file are taken.

- **`num_val_samples` :** Number of samples to use for validation by default all samples in `hdf5` file are taken.

- **`num_tst_samples` :** Number of samples to use for test by default all samples in `hdf5` file are taken.

- **`batch_size` :** Size of each batch given to the **GPUs** or **CPU** depending on what you are training.

- **`num_epochs` :** Number of epochs on which the model will train.

- **`target_size` :**  Sets the crop size of image (H/W) only during training.

- **`loss_fn` :** Loss function, see the documentation on the losses [here](../losses#Losses-available) for all the losses available.

- **`optimizer` :** Optimizer, see the documentation on the optimizers [here](../utils#Optimizers) for all the optimizers available.

- **`learning_rate` :** Initial learning rate.

- **`weight_decay` :** Value for weight decay for each epoch.

- **`step_size` :** Apply gamma for every `step_size`.

- **`gamma` :** Multiple for learning rate decay.

- **`dropout` :** The use dropout or not, it's only applies to certain models, so the default value is `False`.

- **`dropout_prob` :** If `dropout` is `True`, it set dropout probability, example `0.5` for 50%.

- **`class_weights` :** Weights apply to the loss each class (certain loss functions only). We recommend to use weights that ones add to each other will give 1.0, for example, if you have two classes you can weight them `[0.1, 0.9]`

- **`batch_metrics` :** Interval between each batch where all the metrics are computed, if left blank, it will not perform metrics, must be an integer (if `1`, metrics computed on all batches). See the documentation on the metrics [here](../utils#Metrics).

- **`ignore_index` :** Specifies a target value that will be ignored and does not contribute to the input gradient during the training.

- **`normalization` :** The normalization is parameters for fine tuning.
  - **`mean` :** [list of floats] Mean per channel ([link](https://pytorch.org/vision/stable/models.html#semantic-segmentation)).

  - **`std` :** [list of floats] Standard deviation  per channel ([link](https://pytorch.org/vision/stable/models.html#semantic-segmentation)).

- **`augmentation` :** This part is for modifying the images samples to help the network to see different possibilities of representation (also call data augmentation<sup>[1](#dataaugmentation)</sup>). To be perform, all the following parameters need to be feel, otherwise the augmentation will not be performed. For specific details about implementation of these augmentations, check the docstrings in [`augmentation.py`](../utils/augmentation.py).
  - **`rotate_limit` :** The upper and lower limits for data rotation.

  - **`rotate_prob` :** The probability for data rotation.

  - **`hflip_prob` :** The probability for data horizontal flip.

  - **`random_radiom_trim_range` :** The range of the random percentile in which a random percentile value will be chosen to trim values. This value applies to both left and right sides of the raster's histogram.

  - **`brightness_contrast_range` :** # Not yet implemented

  - **`noise` :** # Standard deviation of Gaussian Noise (optional)


#### Running [train_segmentation.py](../train_segmentation.py)

You must run [`images_to_samples.py`](../images_to_samples.py) code before training to generate the `hdf5` use for the training and the validation.

To launch the code:
```shell
python train_segmentation.py path/to/yaml_files/your_config.yaml
```

The output of this code will result at the following structure:
```
├── {data_path}
    └── {samples_folder} (See: the output of the sampling section)
        └── model
            └── {model_name}
                └── checkpoint.pth.tar
                └── {log files}
                └── copy of config.yaml (automatic)
```
The trained model weights will be save as `checkpoint.pth.tar`. Corresponding to the training state where the validation loss was the lowest during the training process. All those information will be stored in the same directory than the `hdf5` images generated by the [sampling](#Sampling).
The `{model_name}` is set from the `yaml` name. Therefore, `yaml` name should be relevant and unique and if the folder already exists, a suffix with `_YYYY-MM-DD_HH-MM` will be added.

> **Advanced features :** To check how a trained model performs on test split without fine-tuning, simply: Specify `training` / `state_dict_path` for this model in `you_config.yaml`, and in same parameter section, set `num_epochs` to 0. The execution will then jump right away to the evaluation on test set with loaded model without training.


## **Inference**
This section is use by [inference.py](../inference.py) to make the inference.This is the final step in the process, assigning every pixel in an image never seen to a value corresponding to the most probable class.
This process will load the trained weights associated to the lower loss score of the chosen model and perform a per-pixel inference task on all the images.

```YAML
inference:
  img_dir_or_csv_file: /path/to/list.csv
  working_folder: /path/to/output_images       
  state_dict_path: /path/to/checkpoint.pth.tar
  chunk_size: 512  
  # Visualization parameters                            
  smooth_prediction: True                      
  overlap: 2                                   
```
- **`img_dir_or_csv_file` :** The path to the `CSV` file containing directory of images with or without `gpkg` labels (used in benchmarking).

- **`working_folder` :** The path to the folder where all resulting images will be written (DEPRECATED, leave blank).

- **`state_dict_path` :** The path to model weights that will be loaded for inference.

- **`chunk_size` :** Size (height and width) of each prediction patch by default it's `512`.

- **`smooth_prediction` :** When `True`, the prediction will be Smoothed with a 2D interpolation.

- **`overlap` :** The overlap between tiles for smoothing. Must be an even number that divides chunk_size without remainder.

#### Running [data_analysis.py](data_analysis.py)
You can run the data analysis alone if you only want the stats, and to do that you only need to launch the program :
```shell
python inference.py path/to/yaml_files/your_config.yaml
```
The output of this code will result at the following structure:
```
├── {state_dict_path}
    └── checkpoint.pth.tar (used for inference)
    └── inference_{num_bands}
        └── {.tifs from inference}
```
The output is compose of one `.tif` per input image with the same dimensions and georeference.


## Visualization
The [visualization](../utils/visualization.py) module is used to visualize the inferred images without loading the `.tif` by saving some `.png` chosen by the `vis_batch_range` parameter.
This code can't be launch alone, can be use in be use in three functions:
1. `vis_from_dataloader()`: iterates through a provided dataloader and sends batch outputs, along with inputs and labels to `vis_from_batch()`. Is used when parameters `vis_at_checkpoint` or `vis_at_init` is `True`.

2. `vis_from_batch()`: iterates though items of a batch and sends them to `vis()`.

3. `vis()`: converts input to 8bit image, iterates through channels of output to extract each heatmap (i.e. activation map) to builds a dictionary with heatmap where key is grayscale value and value is `{'class_name': 'name_of_class', 'heatmap_PIL': heatmap_as_PIL_Image}`. Saves 8bit input, color output and color label (if given) as `.png` in a grid or as individual pngs.

> During inference, visualization functions are also called, but instead of outputting pngs, `vis()` outputs a georeferenced `.tif`.
> Outputs are sent to visualization functions immediately after line `outputs = model(inputs)`, i.e. before `argmax()` function is used to flatten outputs and keep only value to most probable class, pixel-wise.



```yaml
visualization:
  vis_batch_range: [0,200,10]
  vis_at_checkpoint: True    
  vis_at_ckpt_min_ep_diff: 0
  vis_at_ckpt_dataset: val   
  vis_at_init: True          
  vis_at_init_dataset: val
  vis_at_evaluation: True    
  vis_at_train: True         
  grid: True                 
  heatmaps: True             
  colormap_file: ./data/colormap.csv
```
- **`vis_batch_range` :** A list of the *first* batch to visualize on, *last* batch (excluded) and the *increment*. If the list is empty, no visualization will be performed no matter the value of other parameters.

- **`vis_at_checkpoint` :** Visualize samples every time a checkpoint is saved, by default this is `True`.

- **`vis_at_ckpt_min_ep_diff` :** Define the minimum number of epoch that must separate two checkpoints in order to visualize at checkpoint.

- **`vis_at_ckpt_dataset` :** Define dataset to be used for `vis_at_checkpoint`, by default it's `val`.

- **`vis_at_init` :** When `True`, it visualize samples with instantiated model before first epoch.

- **`vis_at_init_dataset` :** Define dataset to be used for `vis_at_init`, by default it's `val`.

- **`vis_at_evaluation` :** When `True`, it visualize samples during evaluation, 'val' during training and 'tst' at end of training.

- **`vis_at_train` :** When `True`, it visualize on training samples during training.

- **`grid` :** When `True`, it save visualization outputs as a grid. If false, each image is saved as separate `.png`.

- **`heatmaps` :** When `True`, it also save heatmaps (activation maps) for each class as given by model before argmax.

- **`colormap_file` (Optional):** custom colormap to define custom class names and colors for visualization, we use `./data/colormap.csv`.

#### Colormap
All visualization functions use a colormap for two main purposes:
1. Assign colors to grayscale outputs (as outputted by pytorch) and labels, if given (i.e. not during inference).

2. Assign a name to each heatmap. This name is displayed above heatmap in grid if `grid: True`. Otherwise, each heatmap is saved as a .png. Class name is then added in the name of .png.
If left empty, a default colormap is used and integers are assigned for each class in output.  

If desired, the user can specify a path to a custom colormap with the `colormap_file` parameter in `your_config.yaml`.
The custom colormap must be a `csv` with 3 columns, as shown below. One line is added for each desired class.

Input value|Class name|Html color
:---:|:---:|:---:
1|vegetation|#00680d
2|hydro|#b2e0e6
3|roads|#990000
4|buildings|#efcd08   


## Final Results
If you run the three code: [images_to_samples.py](../images_to_samples.py), [train_segmentation.py](../train_segmentation.py) and [inference.py](../inference.py) with `your_config.yaml` you should end with a structure like:
```
├── {data_path}
    └── {samples_folder}*
        └── trn_samples.hdf5
        └── val_samples.hdf5
        └── tst_samples.hdf5
        └── model
            └── {model_name}**
                └── checkpoint.pth.tar
                └── {log files}
                └── copy of config.yaml (automatic)
                └── visualization
                    └── {.pngs from visualization}
                └── inference
                    └── {.tifs from inference}
```

<a name="dataaugmentation">1</a>: These augmentations are a [common procedure in machine learning](https://www.coursera.org/lecture/convolutional-neural-networks/data-augmentation-AYzbX). More augmentations could be implemented in a near. See issue [#106](https://github.com/NRCan/geo-deep-learning/issues/106).
