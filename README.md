### Table of Contents
- [Geo-Deep-Learning overview](#Geo-Deep-Learning-overview)
  * [Requirements](#requirements)
  * [Installation on your workstation](#installation-on-your-workstation)
  * [config.yaml](#configyaml)
- [Semantic segmentation](#semantic-segmentation)
    * [Models available](#models-available)
    * [csv preparation](#csv-preparation)
    * [images_to_samples.py](#images_to_samplespy)
    * [train_model.py](#train_modelpy)
    * [inference.py](#inferencepy)
- [Classification Task](#Classification-Task)
    * [Models available](#models-available-1)
    * [Data preparation](#Data-preparation)
    * [train_model.py](#train_modelpy-1)
    * [inference.py](#inferencepy-1)
    
# Geo-Deep-Learning overview

The `geo-deep-learning` project stems from an initiative at NRCan's [CCMEO](https://www.nrcan.gc.ca/earth-sciences/geomatics/10776).  Its aim is to allow using Convolutional Neural Networks (CNN) with georeferenced data sets.

The overall learning process comprises three broad stages: data preparation, training & validation, and inference.  The data preparation phase (sampling) allows creating sub-images that will be used for either training or validation. The training & validation phase learns using the data prepared in the previous phase. Finally, the inference phase allows the use of a trained model on new input data. The training & validation and inference phases currently allow the use of a variety of neural networks to perform classification and semantic segmentation tasks.

> The term `classification` in this project is used as it has been traditionally used in the remote sensing community: a process of assigning land cover classes to pixels.  The meaning of the word in the deep learning community differs somewhat, where classification is simply to assign a label to the whole input image. This usage of the term classification will always be referred to as a ```classification task``` in the context of this project. Other uses of the term classification refer to the final phase of the learning process when a trained model is applied to new images, regardless of whether `semantic segmentation`, ["the process of assigning a label to every pixel in an image"](https://en.wikipedia.org/wiki/Image_segmentation), or a `classification task` is being used.

After installing the required computing environment (see next section), one needs to replace the config.yaml file boilerplate path and other items to point to images and other data.  The full sequence of steps is described in the sections below.

> This project comprises a set of commands to be run at a shell command prompt.  Examples used here are for a bash shell in an Ubuntu GNU/Linux environment.


## Requirements  
- Python 3.6 with the following libraries:
    - pytorch # With your choice of CUDA toolkit
    - torchvision
    - rasterio
    - fiona
    - ruamel_yaml
    - scikit-image
    - scikit-learn
    - h5py
- nvidia GPU highly recommended
- The system can be used on your workstation or cluster and on [AWS](https://aws.amazon.com/).

## Installation on your workstation
1. Using conda, you can set and activate your python environment with the following commands:  
    With GPU (defaults to CUDA 10.0 if `cudatoolkit=X.0` is not specified):
    ```shell
    conda create -p YOUR_PATH python=3.6 pytorch torchvision ruamel_yaml h5py fiona rasterio scikit-image scikit-learn -c pytorch
    source activate YOUR_ENV
    ```
    CPU only:
    ```shell
    conda create -p YOUR_PATH python=3.6 pytorch-cpu torchvision ruamel_yaml h5py fiona rasterio scikit-image scikit-learn -c pytorch
    source activate YOUR_ENV
    ```
1. Set your parameters in the `config.yaml` (see section bellow)
1. Prepare your data and `csv` file
1. Start your task using one of the following command:
    ```shell
    python images_to_samples.py ./conf/config.yaml
    python train_model.py ./conf/config.yaml
    python inference.py ./conf/config.yaml
    ```

## config.yaml

The `config.yaml` file is located in the `conf` directory.  It stores the values of all parameters needed by the deep learning algorithms for all phases.  It is shown below:

```yaml
# Deep learning configuration file ------------------------------------------------
# Five sections:
#   1) Global parameters; those are re-used amongst the next three operations (sampling, training and inference)
#   2) Sampling parameters
#   3) Training parameters
#   4) Inference parameters
#   5) Model parameters

# Global parameters

global:
  samples_size: 256                 # Size (in pixel) of the samples
  num_classes: 2                    # Number of classes
  data_path: /path/to/data/folder   # Path to folder containing samples
  number_of_bands: 3                # Number of bands in input images
  model_name: unetsmall				# One of unet, unetsmall, checkpointed_unet, ternausnet, or inception
  bucket_name:                      # Name of the S3 bucket where data is stored. Leave blank if using local files
  task: segmentation                # Task to perform. Either segmentation or classification.

# Sample parameters; used in images_to_samples.py -------------------

sample:
  prep_csv_file: /path/to/csv/file_name.csv     # Path to CSV file used in preparation.
  samples_dist: 200                             # Distance (in pixel) between samples
  min_annotated_percent: 10                     # Min % of non background pixels in stored samples. Default: 0
  mask_reference: False                         # When True, mask the input image where there is no reference data.

# Training parameters; used in train_model.py ----------------------

training:
  output_path: /path/to/output/weights/folder   # Path to folder where files containing weights will be written
  num_trn_samples: 4960                         # Number of samples to use for training. (default: all samples in hdfs file are taken)
  num_val_samples: 2208                         # Number of samples to use for validation. (default: all samples in hdfs file are taken)
  num_tst_samples:                              # Number of samples to use for test. (default: all samples in hdfs file are taken)
  batch_size: 32                                # Size of each batch
  num_epochs: 150                               # Number of epochs
  learning_rate: 0.0001                         # Initial learning rate
  weight_decay: 0                               # Value for weight decay (each epoch)
  gamma: 0.9                                    # Multiple for learning rate decay
  step_size: 4                                  # Apply gamma every step_size
  class_weights: [1.0, 2.0]                     # Weights to apply to each class. A value > 1.0 will apply more weights to the learning of the class.
  batch_metrics: 2                              # (int) Metrics computed every (int) batches. If left blank, will not perform metrics. If (int)=1, metrics computed on all batches.
  ignore_index: 0                               # Specifies a target value that is ignored and does not contribute to the input gradient. Default: None
  augmentation:
    rotate_limit: 45                            # Specifies the upper and lower limits for data rotation. If not specified, no rotation will be performed.
    rotate_prob: 0.5                            # Specifies the probability for data rotation. If not specified, no rotation will be performed.
    hflip_prob: 0.5                             # Specifies the probability for data horizontal flip. If not specified, no horizontal flip will be performed.
    
# Inference parameters; used in inference.py --------

inference:
  img_csv_file: /path/to/csv/containing/images/list.csv                       # CSV file containing the list of all images to infer on
  working_folder: /path/to/folder/with/resulting/images                       # Folder where all resulting images will be written
  state_dict_path: /path/to/model/weights/for/inference/checkpoint.pth.tar    # File containing pre-trained weights
  chunk_size: 512                                                             # (int) Size (height and width) of each prediction patch. Default: 512
  overlap: 10                                                                 # (int) Percentage of overlap between 2 chunks. Default: 10

# Models parameters; used in train_model.py and inference.py

models:
  unet:   &unet001
    dropout: False
    probability: 0.2    # Set with dropout
    pretrained: False   # optional
  unetsmall:
    <<: *unet001
  ternausnet:
    pretrained: ./models/TernausNet.pt    # Mandatory
  checkpointed_unet:
    <<: *unet001
  inception:
    pretrained: False   # optional
```
# Semantic segmentation
## Models available
- [Unet](https://arxiv.org/abs/1505.04597)
- Unet small (less deep version of Unet)
- Checkpointed Unet (same as Unet small, but uses less GPU memory and recomputes data during the backward pass)
- [Ternausnet](https://arxiv.org/abs/1801.05746)
## `csv` preparation
The `csv` specifies the input images and the reference vector data that will be use during the training.
Each row in the `csv` file must contain 4 comma-separated items:
- input image file (tif)
- reference vector data (GeoPackage)
- attribute of the GeoPackage to use as classes values
- dataset (one of 'trn' for training, 'val' for validation or 'tst' for test) where the sample will be used  

Each image is a new line in the csv file.  For example:  

```
\path\to\input\image1.tif,\path\to\reference\vector1.gpkg,attribute,trn
\path\to\input\image2.tif,\path\to\reference\vector2.gpkg,attribute,val
\path\to\input\image3.tif,\path\to\reference\vector2.gpkg,attribute,tst
```

## images_to_samples.py

The first phase of the process is to determine sub-images (samples) to be used for training, validation and test.  Images to be used must be of the geotiff type.  Sample locations in each image must be stored in a GeoPackage.

To launch the program:  

```
python images_to_samples.py path/to/config/file/config.yaml
```

Details on parameters used by this module:

```yaml
global:
  samples_size: 256                 # Size (in pixel) of the samples
  data_path: /path/to/data/folder   # Path to folder containing samples
  number_of_bands: 3                # Number of bands in input images
  model_name: unetsmall             # One of unet, unetsmall, checkpointed_unet, ternausnet, or inception
  bucket_name:                      # name of the S3 bucket where data is stored. Leave blank if using local files

sample:
  prep_csv_file: /path/to/csv/file_name.csv     # Path to CSV file used in preparation.
  samples_dist: 200                             # Distance (in pixel) between samples
  min_annotated_percent: 10                     # Min % of non background pixels in stored samples. Default: 0
  mask_reference: False                         # When True, mask the input image where there is no reference data.
```

Outputs:
- 3 .hdfs files with input images and reference data, stored as arrays
    - trn_samples.hdfs
    - val_samples.hdfs
    - tst_samples.hdfs

Process:
- Read csv file and for each line in the file, do the following:
    - Create a new raster called "label" with the same properties as the input image
    - Convert GeoPackage vector information into the "label" raster. The pixel value is determined by the attribute in the csv file
    - Convert both input and label images to arrays
    - Divide images in smaller samples of size and distance specified in the configuration file. Visual representation of this is provided [here](https://medium.com/the-downlinq/broad-area-satellite-imagery-semantic-segmentation-basiss-4a7ea2c8466f)
    - Write samples into the "val", "trn" or "tst" hdfs file, depending on the value contained in the csv file.

## train_model.py

The crux of the learning process is in this phase : training.  Samples labeled "trn" as per above are used to train the neural network.  Samples labeled "val" are used to estimate the training error on a set of sub-images not used for training, after every epoch. At the end of all epochs, the model with the lowest error on validation data is loaded and samples labeled "tst" are used to estimate the accuracy of the model on sub-images not used during training or validation.

To launch the program:
```
python train_model.py path/to/config/file/config.yaml
```
Details on parameters used by this module:
```yaml
global:
  samples_size: 256                 # Size (in pixel) of the samples
  num_classes: 2                    # Number of classes
  data_path: /path/to/data/folder   # Path to folder containing samples
  number_of_bands: 3                # Number of bands in input images
  model_name: unetsmall             # One of unet, unetsmall, checkpointed_unet, ternausnet, or inception
  bucket_name:                      # name of the S3 bucket where data is stored. Leave blank if using local files
  task: segmentation                # Task to perform. Either segmentation or classification


training:
  output_path: /path/to/output/weights/folder   # Path to folder where files containing weights will be written
  num_trn_samples: 4960                         # Number of samples to use for training. (default: all samples in hdfs file are taken)
  num_val_samples: 2208                         # Number of samples to use for validation. (default: all samples in hdfs file are taken)
  num_tst_samples:                              # Number of samples to use for test. (default: all samples in hdfs file are taken)
  batch_size: 32                                # Size of each batch
  num_epochs: 150                               # Number of epochs
  learning_rate: 0.0001                         # Initial learning rate
  weight_decay: 0                               # Value for weight decay (each epoch)
  gamma: 0.9                                    # Multiple for learning rate decay
  step_size: 4                                  # Apply gamma every step_size
  class_weights: [1.0, 2.0]                     # Weights to apply to each class. A value > 1.0 will apply more weights to the learning of the class.
  batch_metrics: 2                              # (int) Metrics computed every (int) batches. If left blank, will not perform metrics. If (int)=1, metrics computed on all batches.
  ignore_index: 0                               # Specifies a target value that is ignored and does not contribute to the input gradient. Default: None
  augmentation:
    rotate_limit: 45                            # Specifies the upper and lower limits for data rotation. If not specified, no rotation will be performed.
    rotate_prob: 0.5                            # Specifies the probability for data rotation. If not specified, no rotation will be performed.
    hflip_prob: 0.5                             # Specifies the probability for data horizontal flip. If not specified, no horizontal flip will be performed.    
```

Inputs:
- 1 hdfs file with input images and reference data as arrays used for training (prepared with `images_to_samples.py`)
- 1 hdfs file with input images and reference data as arrays used for validation (prepared with `images_to_samples.py`)
- 1 hdfs file with input images and reference data as arrays used for test (prepared with `images_to_samples.py`)

Output:
- Trained model weights
    - checkpoint.pth.tar        Corresponding to the training state where the validation loss was the lowest during the training process.

Process:
- The application loads the model
- Using the hyperparameters provided in `config.yaml` , the application will try to minimize the cross entropy loss on the training data and evaluate every epoch on the validation data.
- For every epoch, the application shows and log the loss on "trn" and "val" datasets.
- For every epoch, the application shows and log the accuracy, recall and f-score on "val" dataset. Those metrics are also computed on each classes.  
- At the end of the training process, the application shows and log the accuracy, recall and f-score on "tst" dataset. Those metrics are also computed on each classes.  


## inference.py

The final step in the process is to assign very pixel in the original image a value corresponding to the most probable class.

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

inference:
  img_csv_file: /path/to/csv/containing/images/list.csv                       # CSV file containing the list of all images to infer on
  working_folder: /path/to/folder/with/resulting/images                       # Folder where all resulting images will be written
  state_dict_path: /path/to/model/weights/for/inference/checkpoint.pth.tar    # File containing pre-trained weights
  chunk_size: 512                                                             # (int) Size (height and width) of each prediction patch. Default: 512
  overlap: 10                                                                 # (int) Percentage of overlap between 2 chunks. Default: 10
```
Process:
- The process will load trained weights to the chosen model and perform a per-pixel inference task on all the images contained in the working_folder

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


## train_model.py
Samples in the "trn" folder are used to train the model. Samples in the  "val" folder are used to estimate the training error on a set of images not used for training.

During this phase of the classification task, a list of classes is made based on the subfolders in the trn path. The list of classes is saved in a csv file in the same folder as the trained model so that it can be referenced during the classification step.

To launch the program:
```
python train_model.py path/to/config/file/config.yaml
```
Details on parameters used by this module:
```yaml
global:
  data_path: /path/to/data/folder   # Path to folder containing samples
  number_of_bands: 3                # Number of bands in input images
  model_name: inception             # One of unet, unetsmall, checkpointed_unet, ternausnet, or inception
  bucket_name:                      # name of the S3 bucket where data is stored. Leave blank if using local files
  task: classification               # Task to perform. Either segmentation or classification

training:
  output_path: /path/to/output/weights/folder   # Path to folder where files containing weights will be written
  batch_size: 32                                # Size of each batch
  num_epochs: 150                               # Number of epochs
  learning_rate: 0.0001                         # Initial learning rate
  weight_decay: 0                               # Value for weight decay (each epoch)
  gamma: 0.9                                    # Multiple for learning rate decay
  step_size: 4                                  # Apply gamma every step_size
  class_weights: [1.0, 2.0]                     # Weights to apply to each class. A value > 1.0 will apply more weights to the learning of the class.
  batch_metrics: 2                              # (int) Metrics computed every (int) batches. If left blank, will not perform metrics. If (int)=1, metrics computed on all batches.
```
Note: ```data_path``` must always have a value for classification tasks

Inputs:
- Tiff images in the file structure described in the Classification Task Data Preparation section

Output:
- Trained model weights
    - checkpoint.pth.tar        Corresponding to the training state where the validation loss was the lowest during the training process.
    - last_epoch.pth.tar         Corresponding to the training state after the last epoch.

Process:
- The application loads the model specified in the configuration file
- Using the hyperparameters provided in `config.yaml` , the application will try to minimize the cross entropy loss on the training and validation data
- For every epoch, the application shows the loss, accuracy, recall and f-score for both datasets (trn and val)
- The application also log the accuracy, recall and f-score for each classes of both the datasets


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
  task: classification        # Task to perform. Either segmentation or classification

inference:
  img_csv_file: /path/to/csv/containing/images/list.csv                       # CSV file containing the list of all images to infer on
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
