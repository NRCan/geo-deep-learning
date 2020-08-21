### Table of Contents
- [Geo-Deep-Learning overview](#Geo-Deep-Learning-overview)
  * [Requirements](#requirements)
  * [Installation on your workstation](#installation-on-your-workstation)
  * [config.yaml](#configyaml)
- [Semantic segmentation](#semantic-segmentation)
    * [Folder structure](#folder-structure)
    * [Models available](#models-available)
    * [csv preparation](#csv-preparation)
    * [images_to_samples.py](#images_to_samplespy)
    * [train_segmentation.py](#train_segmentationpy)
    * [inference.py](#inferencepy)
    * [visualization](#visualization)
- [Classification Task](#Classification-Task)
    * [Models available](#models-available-1)
    * [Data preparation](#Data-preparation)
    * [train_classification.py](#train_classificationpy-1)
    * [inference.py](#inferencepy-1)
    
# Geo-Deep-Learning overview

The `geo-deep-learning` project stems from an initiative at NRCan's [CCMEO](https://www.nrcan.gc.ca/earth-sciences/geomatics/10776).  Its aim is to allow using Convolutional Neural Networks (CNN) with georeferenced data sets.

The overall learning process comprises three broad stages: (1) data preparation, (2) training (along with validation and testing), and (3) inference.  The data preparation phase (sampling) allows creating sub-images that will be used for either training, validation or testing. The training phase learns using the data prepared in the previous phase. Finally, the inference phase allows the use of a trained model on new input data. The training and inference phases currently allow the use of a variety of neural networks to perform classification and semantic segmentation tasks.

> The term `classification` in this project is used as it has been traditionally used in the remote sensing community: a process of assigning land cover classes to pixels.  The meaning of the word in the deep learning community differs somewhat, where classification is simply to assign a label to the whole input image. This usage of the term classification will always be referred to as a ```classification task``` in the context of this project. Other uses of the term classification refer to the final phase of the learning process when a trained model is applied to new images, regardless of whether `semantic segmentation`, ["the process of assigning a label to every pixel in an image"](https://en.wikipedia.org/wiki/Image_segmentation), or a `classification task` is being used.

After installing the required computing environment (see next section), one needs to replace the config.yaml file boilerplate path and other items to point to images and other data.  The full sequence of steps is described in the sections below.

> This project comprises a set of commands to be run at a shell command prompt.  Examples used here are for a bash shell in an Ubuntu GNU/Linux environment.

## Requirements  
- Python 3.6 with the following libraries:
    - pytorch # With your choice of CUDA toolkit
    - torchvision
    - opencv
    - rasterio
    - fiona
    - ruamel_yaml
    - scikit-image
    - scikit-learn
    - h5py
    - nvidia-ml-py3
    - tqdm
- nvidia GPU highly recommended
- The system can be used on your workstation or cluster and on [AWS](https://aws.amazon.com/).

## Installation on your workstation using miniconda
1. Using conda, you can set and activate your python environment with the following commands:  
    With GPU (defaults to CUDA 10.0 if `cudatoolkit=X.0` is not specified):
    ```shell
    conda create -n gpu_ENV python=3.6 -c pytorch pytorch torchvision 
    conda activate gpu_ENV
    conda install -c conda-forge ruamel_yaml h5py fiona rasterio geopandas scikit-image scikit-learn tqdm 
    conda install -c fastai nvidia-ml-py3
    conda install mlflow 
    ```
    CPU only:
    ```shell
    conda create -n cpu_ENV python=3.6 -c pytorch pytorch-cpu torchvision-cpu 
    conda activate cpu_ENV
    conda install -c conda-forge opencv
    conda install -c conda-forge ruamel_yaml h5py fiona rasterio geopandas scikit-image scikit-learn tqdm
    conda install mlflow 
    ```
    > For Windows OS: 
    > - Install rasterio, fiona and gdal first, before installing the rest. We've experienced some [installation issues](https://github.com/conda-forge/gdal-feedstock/issues/213), with those libraries. 
    > - Mlflow should be installed using pip rather than conda, as mentionned [here](https://github.com/mlflow/mlflow/issues/1951)  
    >
1. Set your parameters in the `config.yaml` (see section below)
1. Prepare your data and `csv` file
1. Start your task using one of the following command:
    ```shell
    python images_to_samples.py ./conf/config.yaml
    python train_segmentation.py ./conf/config.yaml
    python inference.py ./conf/config.yaml
    ```

## config.yaml

The `config.yaml` file is located in the `conf` directory.  It stores the values of all parameters needed by the deep learning algorithms for all phases.  It contains the following 4 sections:

```yaml
# Deep learning configuration file ------------------------------------------------
# Five sections:
#   1) Global parameters; those are re-used amongst the next three operations (sampling, training and inference)
#   2) Sampling parameters
#   3) Training parameters
#   4) Inference parameters
#   5) Visualization

```

Specific parameters in each section are shown below, where relevant. For more information about config.yaml, view file directly: [conf/config.yaml](https://github.com/NRCan/geo-deep-learning/blob/master/conf/config.yaml)

# Semantic segmentation

## Folder Structure

Suggested high level structure:
```
├── {dataset_name}
    └── data
        └── RGB_tiff
            └── {3 band tiff images}
        └── RGBN_tiff
            └── {4 band tiff images}
        └── gpkg
            └── {GeoPackages}
        └── {csv_samples_preparation}.csv
        └── yaml_files
            └── {yaml config files}
        └── {data_path} (see below)
├── geo-deep-learning
    └── {scripts as cloned from github}
```

Structure as created by geo-deep-learning
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
*See: [images_to_samples.py / Outputs](samples_outputs)

**See: [train_segmentation.py / Outputs](training_outputs)

## Models available
- [Unet](https://arxiv.org/abs/1505.04597)
- [Deeplabv3 (backbone: resnet101, optional: pretrained on coco dataset)](https://arxiv.org/abs/1706.05587)
- Experimental: Deeplabv3 (default: pretrained on coco dataset) adapted for RGB-NIR(4 Bands) supported
- Unet small (less deep version of Unet)
- Checkpointed Unet (same as Unet small, but uses less GPU memory and recomputes data during the backward pass)
- [Ternausnet](https://arxiv.org/abs/1801.05746)
- [FCN (backbone: resnet101, optional: pretrained on coco dataset)](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)

## `csv` preparation
The `csv` specifies the input images and the reference vector data that will be use during the training.
Each row in the `csv` file must contain 5 comma-separated items:
- input image file (tif)
- metadata info (*more details to come*). Optional, leave empty if not desired. (see example below)
- reference vector data (GeoPackage)
- attribute of the GeoPackage to use as classes values
- dataset (one of 'trn' for training, 'val' for validation or 'tst' for test) where the sample will be used. Test set is optional.  

Each image is a new line in the csv file.  For example:  

```
\path\to\input\image1.tif,,\path\to\reference\vector1.gpkg,attribute,trn
\path\to\input\image2.tif,,\path\to\reference\vector2.gpkg,attribute,val
\path\to\input\image3.tif,,\path\to\reference\vector2.gpkg,attribute,tst
```

## images_to_samples.py

The first phase of the process is to determine sub-images (samples) to be used for training, validation and, optionally, test.  Images to be used must be of the geotiff type.  Sample locations in each image must be stored in a GeoPackage.

To launch the program:  

```
python images_to_samples.py path/to/config/file/config.yaml
```

> Note: A data analysis module can be found [here](./docs/data_analysis.md).  
> It is useful for balancing training data.

Details on parameters used by this module:

```yaml
global:
  samples_size: 256         # Size (in pixel) of the samples.
  num_classes: 2            # Number of classes. 
  data_path: /path/to/data  # Path to folder where samples folder will be automatically created
  number_of_bands: 3        # Number of bands in input images.
  model_name: unetsmall     # One of unet, unetsmall, checkpointed_unet, ternausnet, or inception
  bucket_name:              # name of the S3 bucket where data is stored. Leave blank if using local files
  scale_data: [0, 1]        # Min and Max for input data rescaling. Default: [0, 1]. Default: No rescaling
  debug_mode: True          # Activates various debug features (ex.: details about intermediate outputs, detailled progress bars, etc.). Default: False

sample:
  prep_csv_file: /path/to/file_name.csv  # Path to CSV file used in preparation.
  overlap: 200                           # (int) Percentage of overlap between 2 samples. Mandatory
  val_percent: 5                         # Percentage of validation samples created from train set (0 - 100)
  min_annotated_percent: 10              # Min % of non background pixels in stored samples. Mandatory
  mask_reference: False                  # When True, mask the input image where there is no reference data.
```

### Process
1. Read csv file and validate existence of all input files and GeoPackages.
2. Do the following verifications:
    1. Assert number of bands found in raster is equal to desired number of bands 
    2. Check that `num_classes` is equal to number of classes detected in the specified attribute for each GeoPackage. Warning: this validation **will not succeed** if a Geopackage contains only a subset of `num_classes` (e.g. 3 of 4).
    3. Assert Coordinate reference system between raster and gpkg match. 
3. Read csv file and for each line in the file, do the following:
    1. Read input image as array with `utils.readers.image_reader_as_array()`.
        - If gpkg's extent is smaller than raster's extent, raster is clipped to gpkg's extent.
        - If gpkg's extent is bigger than raster's extent, gpkg is clipped to raster's extent. 
    2. Convert GeoPackage vector information into the "label" raster with `utils.utils.vector_to_raster()`. The pixel value is determined by the attribute in the csv file.
    3. Create a new raster called "label" with the same properties as the input image
    4. Read metadata and add to input as new bands (*more details to come*)
    5. Crop arrays in smaller samples of size `samples_size` and distance `num_classes` specified in the configuration file. Visual representation of this is provided [here](https://medium.com/the-downlinq/broad-area-satellite-imagery-semantic-segmentation-basiss-4a7ea2c8466f)
    6. Write samples from input image and label into the "val", "trn" or "tst" hdf5 file, depending on the value contained in the csv file. Refer to `samples_preparation()`. 

### <a name="samples_outputs"></a> Outputs
- 3 .hdf5 files with input images and reference data, stored as arrays, with following structure:
```
├── {data_path}
    └── {samples_folder}* 
        └── trn_samples.hdf5
        └── val_samples.hdf5
        └── tst_samples.hdf5
```
*{samples_folder} is set from values in .yaml: 

"samples{`samples_size`}\_overlap{`overlap`}\_min-annot{`min_annot_perc`}\_{`num_bands`}bands" 

>If folder already exists, a suffix with `_YYYY-MM-DD_HH-MM` is added

### Debug mode
- Images_to_samples.py will assert that all geometries for features in GeoPackages are valid according to [Rasterio's algorithm](https://github.com/mapbox/rasterio/blob/d4e13f4ba43d0f686b6f4eaa796562a8a4c7e1ee/rasterio/features.py#L461).   

## train_segmentation.py

The crux of the learning process is in this phase : training.  
- Samples labeled "trn" as per above are used to train the neural network.
- Samples labeled "val" are used to estimate the training error (i.e. loss) on a set of sub-images not used for training, after every epoch. 
- At the end of all epochs, the model with the lowest error on validation data is loaded and samples labeled "tst", if they exist, are used to estimate the accuracy of the model on sub-images unseen during training or validation.

To launch the program:
```
python train_segmentation.py path/to/config/file/config.yaml
```
Details on parameters used by this module:
```yaml
global:
  samples_size: 256          # Size (in pixel) of the samples
  num_classes: 2             # Number of classes
  data_path: /path/to/data   # Path to folder containing samples folder. Model and log files will be written in samples folder
  number_of_bands: 3         # Number of bands in input images
  model_name: unetsmall      # One of unet, unetsmall, checkpointed_unet, ternausnet, or inception
  bucket_name:               # name of the S3 bucket where data is stored. Leave blank if using local files
  task: segmentation         # Task to perform. Either segmentation or classification
  num_gpus: 0                # Number of GPU device(s) to use. Default: 0
  debug_mode: True           # Activates various debug features (ex.: details about intermediate outputs, detailled progress bars, etc.). Default: False

sample:
  overlap: 20                # % of overlap between 2 samples.
  min_annotated_percent: 10  # Min % of non background pixels in stored samples.

training:
  state_dict_path: /path/to/checkpoint.pth.tar  # path to checkpoint from trained model as .pth.tar or .pth file. Optional.
  pretrained: True           # if True, pretrained model will be loaded if available (e.g. Deeplabv3 pretrained on coco dataset). Default: True if no state_dict is given
  num_trn_samples: 4960      # Number of samples to use for training. (default: all samples in hdfs file are taken)
  num_val_samples: 2208      # Number of samples to use for validation. (default: all samples in hdfs file are taken)
  num_tst_samples:           # Number of samples to use for test. (default: all samples in hdfs file are taken)
  batch_size: 32             # Size of each batch
  num_epochs: 150            # Number of epochs
  loss_fn: Lovasz            # One of CrossEntropy, Lovasz, Focal, OhemCrossEntropy (*Lovasz for segmentation tasks only)
  optimizer: adabound        # One of adam, sgd or adabound
  learning_rate: 0.0001      # Initial learning rate
  weight_decay: 0            # Value for weight decay (each epoch)
  step_size: 4               # Apply gamma every step_size
  gamma: 0.9                 # Multiple for learning rate decay
  dropout: False             # (bool) Use dropout or not. Applies to certain models only.
  dropout_prob: False        # (float) Set dropout probability, e.g. 0.5
  class_weights: [1.0, 2.0]  # Weights to apply to each class. A value > 1.0 will apply more weights to the learning of the class. Applies to certain loss functions only.
  batch_metrics: 2           # (int) Metrics computed every (int) batches. If left blank, will not perform metrics. If (int)=1, metrics computed on all batches.
  ignore_index: 0            # Specifies a target value that is ignored and does not contribute to the input gradient. Default: None
```

### Inputs
- samples folder as created by `images_to_samples.py` (See: [Images_to_samples.py / Outputs](#samples_outputs)) containing:
    - `trn_samples.hdf5`, `val_samples.hdf5`, `tst_samples.hdf5`. Each hdf5 file contains input images and reference data as arrays used for training, validation and test, respectively.
    
#### Augmentations

Details on parameters used by this module:
```yaml  
training:
   augmentation:
        rotate_limit: 45         # Specifies the upper and lower limits for data rotation. If not specified, no rotation will be performed.
        rotate_prob: 0.5         # Specifies the probability for data rotation. If not specified, no rotation will be performed.
        hflip_prob: 0.5          # Specifies the probability for data horizontal flip. If not specified, no horizontal flip will be performed.    
        random_radiom_trim_range: [0.1, 2.0] # Specifies the range in which a random percentile value will be chosen to trim values. This value applies to both left and right sides of the raster's histogram. If not specified, no enhancement will be performed. 
        geom_scale_range:        # Not yet implemented
        noise:                   # Not yet implemented
```
These augmentations are a [common procedure in machine learning](https://www.coursera.org/lecture/convolutional-neural-networks/data-augmentation-AYzbX). More augmentations could be implemented in a near. See issue [#106](https://github.com/NRCan/geo-deep-learning/issues/106). 

Note: For specific details about implementation of these augmentations, check docstrings in utils.augmentation.py. 
 
Warning: 
- RandomCrop is used only if parameter `target_size` (in training section of config file) is not empty. Also, if this parameter is omitted, the hdf5 samples will be fed as is to the model in training (after other augmentations, if applicable). This can have an major impact on GPU RAM used and could cause `CUDA: Out of memory errors`.  

### Process
1. Model is instantiated and checkpoint is loaded from path, if provided in `config.yaml`.
2. GPUs are requested according to desired amount of `num_gpus` and available GPUs.
3. If more than 1 GPU is requested, model is casted to [`DataParallel`](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html) model
4. Dataloaders are created with `create_dataloader()`
5. Loss criterion, optimizer and learning rate are set with `set_hyperparameters()` as requested in `config.yaml`.
5. Using these hyperparameters, the application will try to minimize the loss on the training data and evaluate every epoch on the validation data.
6. For every epoch, the application shows and logs the loss on "trn" and "val" datasets.
7. For every epoch (if `batch_metrics: 1`), the application shows and logs the accuracy, recall and f-score on "val" dataset. Those metrics are also computed on each classes.  
8. At the end of the training process, the application shows and logs the accuracy, recall and f-score on "tst" dataset. Those metrics are also computed on each classes.

### <a name="training_outputs"></a> Output
- Trained model weights as `checkpoint.pth.tar`. Corresponding to the training state where the validation loss was the lowest during the training process.

```
├── {data_path}
    └── {samples_folder} (See: images_to_samples.py / Outputs) 
        └── model
            └── {model_name}*
                └── checkpoint.pth.tar
                └── {log files}
                └── copy of config.yaml (automatic)
```
*{model_name} is set from yaml name. Therefore, **yaml name should be relevant and unique**. If folder already exists, a suffix with `_YYYY-MM-DD_HH-MM` is added.

### Loss functions
- Cross-Entropy (standard loss functions as implemented in [torch.nn](https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html))
- [Multi-class Lovasz-Softmax loss](https://arxiv.org/abs/1705.08790)
- Ohem Cross Entropy. Adapted from [OCNet Repository](https://github.com/PkuRainBow/OCNet)
- [Focal Loss](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938) 

### Optimizers
- Adam (standard optimizer in [torch.optim](https://pytorch.org/docs/stable/optim.html))
- SGD (standard optimizer in [torch.optim](https://pytorch.org/docs/stable/optim.html)
- [Adabound/AdaboundW](https://openreview.net/forum?id=Bkg3g2R9FX)

### Debug mode
- During training, `train_segmentation.py` will display in progress bar:
    - train loss for last batch 
    - GPUs usage (% and RAM)
    - current learning rate
    - input image shape
    - label shape
    - batch size
    - number of predicted classes in each output array (if only one predicted class, there's a problem!)   

### Advanced features
- To check how a trained model performs on test split without fine-tuning, simply:
    1. Specify `training` / `state_dict_path` for this model in `config.yaml` 
    2. In same parameter section, set `num_epochs` to 0. The execution will then jump right away to `evaluation` on test set with loaded model without training.

## inference.py

The final step in the process is to assign every pixel in the original image a value corresponding to the most probable class.

To launch the program:
```
python inference.py path/to/config/file/config.yaml
```

Details on parameters used by this module:
```yaml
global:
  number_of_bands: 3        # Number of bands in input images
  model_name: unetsmall     # One of unet, unetsmall, checkpointed_unet, ternausnet, or inception
  bucket_name:              # name of the S3 bucket where data is stored. Leave blank if using local files
  task: segmentation        # Task to perform. Either segmentation or classification
  scale_data: [0, 1]        # Min and Max for input data rescaling. Default: [0, 1]. Enter False if no rescaling is desired.
  debug_mode: True          # Prints detailed progress bar


inference:
  img_dir_or_csv_file: /path/to/list.csv        # CSV file containing directory of images with or without gpkg labels(used in benchmarking) 
  working_folder: /path/to/output_images        # Folder where all resulting images will be written (DEPRECATED, leave blank)
  state_dict_path: /path/to/checkpoint.pth.tar  # Path to model weights for inference
  chunk_size: 512                               # (int) Size (height and width) of each prediction patch. Default: 512
  smooth_prediction: True                       # Smoothening Predictions with 2D interpolation
  overlap: 2                                    # overlap between tiles for smoothing. Must be an even number that divides chunk_size without remainder.
```
### Process
- The process will load trained weights to the chosen model and perform a per-pixel inference task on all the images contained in the working_folder

### Outputs
- one .tif per input image. Output file has same dimensions as input and georeference.
- Structure: 
```
├── {state_dict_path}
    └── checkpoint.pth.tar (used for inference)
    └── inference_{num_bands}
        └── {.tifs from inference}
```

### Debug mode
- During inference, visualization is performed for each inferred chunk
- Detailled progress bar with: 
    - GPUs usage (% and RAM)
    - input chunk shape
    - output shape
    - overlay between chunks  
- output_counts.png is saved. Let's user see regions were multiple inferences are done.

## Visualization
 
Details on parameters used by this module:
```yaml
visualization:
  vis_batch_range: [0,200,10] #first batch to visualize on, last batch (excluded), increment. If empty, no visualization will be performed no matter the value of other parameters.
  vis_at_checkpoint: True     # Visualize samples every time a checkpoint is saved 
  vis_at_ckpt_min_ep_diff: 0  # Define minimum number of epoch that must separate two checkpoints in order to visualize at checkpoint 
  vis_at_ckpt_dataset: val    # define dataset to be used for vis_at_checkpoint. Default: 'val'
  vis_at_init: True           # Visualize samples with instantiated model before first epoch
  vis_at_init_dataset: val    # define dataset to be used for vis_at_init. Default: 'val'
  vis_at_evaluation: True     # Visualize samples val during evaluation, 'val' during training, 'tst' at end of training
  vis_at_train: True          # Visualize on training samples during training
  grid: True                  # Save visualization outputs as a grid. If false, each image is saved as separate .png. Default: False
  heatmaps: True              # Also save heatmaps (activation maps) for each class as given by model before argmax.
  colormap_file: ./data/colormap.csv # Custom colormap to define custom class names and colors for visualization. Optional
```

### Colormap

All visualization functions use a colormap for two main purposes: 
1. Assign colors to grayscale outputs (as outputted by pytorch) and labels, if given (i.e. not during inference).
2. Assign a name to each heatmap. This name is displayed above heatmap in grid if `grid: True`. Otherwise, each heatmap is saved as a .png. Class name is then added in the name of .png.
If left empty, a default colormap is used and integers are assigned for each class in output.  

If desired, the user can therefore specify a path to a custom colormap with the `colormap_file` parameter in the `config.yaml`.
The custom colormap must be a .csv with 3 columns, as shown below. One line is added for each desired class.

Input grayscale value|Class name|Html color (#RRGGBB)
---|---|---
1|vegetation|#00680d
2|hydro|#b2e0e6
3|roads|#990000
4|buildings|#efcd08   

### Process and Outputs
Visualization is called in three main functions:
1. `vis_from_dataloader()`: iterates through a provided dataloader and sends batch outputs, along with inputs and labels to `vis_from_batch()`. Is used when parameters `vis_at_checkpoint` or `vis_at_init` is `True`
2. `vis_from_batch()`: iterates though items of a batch and sends them to `vis()` 
3. `vis()`: 
    1. converts input to 8bit image if scaling had been performed during training (e.g. scaling between 0 and 1).
    2. iterates through channels of output to extract each heatmap (i.e. activation map)
    3. builds dictionary with heatmap where key is grayscale value and value is `{'class_name': 'name_of_class', 'heatmap_PIL': heatmap_as_PIL_Image}` 
    4. saves 8bit input, color output and color label (if given) as .png in a grid or as individual pngs.
 
The `vis_batch_range` parameter plays a central role in visualization. First number in list is first batch to visualize on. Second number is last batch (excluded) from which no more visualization is done. Last number in list is increment in batch index. For example, if `vis_batch_range = [10,20,2]`, visualization will occur (as requested by other visualization parameters) on batches 10, 12, 14, 16 et 18. **If `vis_batch_range` is left empty, no visualization will be performed no matter the value of other parameters.**

> Outputs are sent to visualization functions immediately after line `outputs = model(inputs)`, i.e. before `argmax()` function is used to flatten outputs and keep only value to most probable class, pixel-wise.

> During inference, visualization functions are also called, but instead of outputting .pngs, `vis()` outputs a georeferenced .tif. Heatmaps, if `inference`/`heatmaps` is `True`, are also saved as georeferenced .tifs, in grayscale format (i.e. single band).
 
### Debug mode
- if in inference, `vis()` will print all unique values in each heatmap array. If there are only a few values, it gives a hint on usefulness of heatmap.
- if in inference, `vis()` will check number of predicted classes in output array. If only one predicted class, a warning is sent.

# Classification Task
The classification task allows images to be recognized as a whole rather than identifying the class of each pixel individually as is done in semantic segmentation.

Currently, Inception-v3 is the only model available for classification tasks in our deep learning process. Other model architectures may be added in the future.

## Models available
- [Inception-v3](https://arxiv.org/abs/1512.00567)
## Data preparation
The images used for training the model must be split into folders for training and validation samples within the ```data_path``` global parameter from the configuration file. Each of these folders must be divided into subfolders by class in a structure like ImageNet-like structure. Torchvision's ```ImageLoader``` is used as the dataset for training and thus running ```images_to_samples.py``` isn't necessary when performing classification tasks. An example of the required file structure is provided below:

```
data_path
├── trn
│   ├── grassland
│   │   ├── 103.tif
│   │   └── 99.tif
│   ├── roads
│   │   ├── 1018.tif
│   │   └── 999.tif
│   ├── trees
│   │   ├── 1.tif
│   │   └── 94.tif
│   └── water
│       ├── 100.tif
│       └── 98.tif
└── val
    ├── building
    │   └── 323955.tif
    ├── grassland
    │   ├── 323831.tif
    │   └── 323999.tif
    ├── roads
    │   └── 323859.tif
    ├── trees
    │   └── 323992.tif
    └── water
        └── 323998.tif
```


## train_classification.py
Samples in the "trn" folder are used to train the model. Samples in the  "val" folder are used to estimate the training error on a set of images not used for training.

During this phase of the classification task, a list of classes is made based on the subfolders in the trn path. The list of classes is saved in a csv file in the same folder as the trained model so that it can be referenced during the classification step.

To launch the program:
```
python train_classification.py path/to/config/file/config.yaml
```
Details on parameters used by this module:
```yaml
global:
  data_path: /path/to/data/folder   # Path to folder containing samples
  number_of_bands: 3                # Number of bands in input images
  model_name: inception             # One of unet, unetsmall, checkpointed_unet, ternausnet, or inception
  bucket_name:                      # name of the S3 bucket where data is stored. Leave blank if using local files
  task: classification              # Task to perform. Either segmentation or classification
  debug_mode: True                  # Prints detailed progress bar with sample loss, GPU stats (RAM, % of use) and information about current samples.

training:
  state_dict_path: False      # Pretrained model path as .pth.tar or .pth file. Optional.
  batch_size: 32                                # Size of each batch
  num_epochs: 150                               # Number of epochs
  learning_rate: 0.0001                         # Initial learning rate
  weight_decay: 0                               # Value for weight decay (each epoch)
  step_size: 4                                  # Apply gamma every step_size
  gamma: 0.9                                    # Multiple for learning rate decay
  dropout: False                                # (bool) Use dropout or not. Applies to certain models only.
  dropout_prob: False                           # (float) Set dropout probability, e.g. 0.5
  class_weights: [1.0, 2.0]                     # Weights to apply to each class. A value > 1.0 will apply more weights to the learning of the class.
  batch_metrics: 2                              # (int) Metrics computed every (int) batches. If left blank, will not perform metrics. If (int)=1, metrics computed on all batches.
  ignore_index: 0                               # Specifies a target value that is ignored and does not contribute to the input gradient. Default: None
  augmentation:
    rotate_limit: 45
    rotate_prob: 0.5
    hflip_prob: 0.5
```
Note: ```data_path``` must always have a value for classification tasks

Inputs:
- Tiff images in the file structure described in the Classification Task Data Preparation section

Output:
- Trained model weights
    - checkpoint.pth.tar        Corresponding to the training state where the validation loss was the lowest during the training process.
    - last_epoch.pth.tar         Corresponding to the training state after the last epoch.
- Model weights and log files are saved to: data_path / 'model' / name_of_.yaml_file.
- If running multiple tests with same data_path, a suffix containing date and time is added to directory (i.e. name of .yaml file)

Process:
- The application loads the model specified in the configuration file
- Using the hyperparameters provided in `config.yaml` , the application will try to minimize the cross entropy loss on the training and validation data
- For every epoch, the application shows the loss, accuracy, recall and f-score for both datasets (trn and val)
- The application also log the accuracy, recall and f-score for each classes of both the datasets

Loss functions:
- Cross-Entropy (standard loss functions as implemented in [torch.nn](https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html))
- Ohem Cross Entropy. Adapted from [OCNet Repository](https://github.com/PkuRainBow/OCNet)
- [Focal Loss](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938) 

Optimizers:
- Adam (standard optimizer in [torch.optim](https://pytorch.org/docs/stable/optim.html))
- SGD (standard optimizer in [torch.optim](https://pytorch.org/docs/stable/optim.html)
- [Adabound/AdaboundW](https://openreview.net/forum?id=Bkg3g2R9FX)

## inference.py
The final step of a classification task is to associate a label to each image that needs to be classified. The associations will be displayed on the screen and be saved in a csv file.

The classes.csv file must be saved in the same folder as the trained model weights file.

To launch the program:
```
python inference.py path/to/config/file/config.yaml
```

Details on parameters used by this module:
```yaml
global:
  number_of_bands: 3        # Number of bands in input images
  model_name: inception     # One of unet, unetsmall, checkpointed_unet or ternausnet
  bucket_name:              # name of the S3 bucket where data is stored. Leave blank if using local files
  task: classification      # Task to perform. Either segmentation or classification
  debug_mode: True          # Prints detailed progress bar

inference:
  img_dir_or_csv_file: /path/to/csv/containing/images/list.csv                 # Directory containing all images to infer on OR CSV file with list of images
  working_folder: /path/to/folder/with/resulting/images                       # Folder where all resulting images will be written
  state_dict_path: /path/to/model/weights/for/inference/checkpoint.pth.tar    # File containing pre-trained weights
```
Inputs:
- Trained model (weights)
- csv with the list of classes used in training
- Images to be classified

Outputs:
- csv file associating each image by its file path to a label. This file also contains the class prediction vector with the classes in the same order as in classes.csv if it was generated during training.

Process:
- The process will load trained weights to the specified architecture and perform a classification task on all the images contained in the ```working_folder```.
- The full file path of the classified image, the class identified, as well as the top 5 most likely classes and their value will be displayed on the screen
