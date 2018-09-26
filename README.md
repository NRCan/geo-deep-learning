# geo-deep-learning

The `geo-deep-learning` project stems from an initiative at NRCan's [CCMEO](https://www.nrcan.gc.ca/earth-sciences/geomatics/10776).  Its aim is to allow using Convolutional Neural Networks (CNN) with georeferenced data sets.

The overall learning process comprises three broad phases : data preparation, training & validation, and classification.  The data preparation phase (sampling) allows creating sub-images that will be used for either training or validation. The training & validation phase learns using the data prepared in the previous phase. Finally, the classification phase performs the classification on new input data. The training & validation and classification phases currently allow the use of a UNET neural network to perform the classification. In future code iterations, we would like to parametrize the choice of neural network type and enable a capacity for other computer vision tasks. 

> The term `classification` in this project is used like it has been traditionally used in the remote sensing community: a process of assigning land cover classes to pixels.  The meaning of the word in the deep learning community differs somewhat, where classification is simply to assign a label to the whole input image.  In that world, classification as it is understood here is expressed as `semantic segmentation`, ["the process of assigning a label to every pixel in an image"](https://en.wikipedia.org/wiki/Image_segmentation).

After installing the required computing environment (see next section), one needs to replace the config.yaml file boilerplate path and other items to point to images and other data.  The full sequence of steps is described in the sections below. 

> This project comprises a set of commands to be run at a shell command prompt.  Examples used here are for a bash shell in a Ubuntu GNU/Linux environment.

## Requirements  
- Python 3.6 with the following libraries:
    - pytorch 0.4.0
    - torchvision 0.4.0
    - numpy 1.14.0
    - gdal 2.2.2
    - ruamel_yaml 0.15.35
    - scikit-image 0.13.1
    - scikit-learn 0.20 (version on [conda-forge](https://anaconda.org/conda-forge/scikit-learn) as of 26 Sept. 2018)
    - h5py 2.8.0
- nvidia GPU highly recommended

Installation using conda can be performed with the following commands:

```shell
conda create -p YOUR_PATH python=3.6 pytorch=0.4.0 torchvision cuda80 ruamel_yaml h5py gdal=2.2.2 scikit-image -c pytorch  
pip install scikit-learn==0.20 # Until scikit-learn 0.20 makes it to the main Anaconda repo
```

> `scikit-learn` is used to output statistics.  The code in this project is compatible with the next version of scikit-learn, which is currently available only as a Release Candidate.

## config.yaml

The `config.yaml` file is located in the `conf` directory.  It stores the values of all parameters needed by the deep learning algorithms for all phases.  It is shown below: 

```yaml
# Deep learning configuration file ------------------------------------------------
# Four sections :
#   1) Global parameters; those are re-used amongst the next three operations (sampling, training and classification)
#   2) Sampling parameters
#   3) training parameters
#   4) Classification parameters

# Global parameters

global:
  samples_size: 256                 # Size (in pixel) of the samples
  num_classes: 2                    # Number of classes
  data_path: /path/to/data/folder   # Path to folder containing samples
  number_of_bands: 3                # Number of bands in input images

# Sample parameters; used in images_to_samples.py -------------------

sample:
  prep_csv_file: /path/to/csv/file_name.csv     # Path to CSV file used in preparation.
  samples_dist: 200                             # Distance (in pixel) between samples
  remove_background: True                       # When True, does not write samples containing only "0" values.
  mask_input_image: False                       # When True, mask the input image where there is no reference data.

# Training parameters; used in train_model.py ----------------------

training:
  output_path: /path/to/output/weights/folder   # Path to folder where files containing weights will be written
  num_trn_samples: 4960                         # Number of samples to use for training (should be a multiple of batch_size)
  num_val_samples: 2208                         # Number of samples to use for validation (should be a multiple of batch_size)
  pretrained: False                             # .pth.tar filename containig pre-trained weights
  batch_size: 32                                # Size of each batch
  num_epochs: 150                               # Number of epochs
  learning_rate: 0.0001                         # Initial learning rate
  weight_decay: 0                               # Value for weight decay (each epoch)
  gamma: 0.9                                    # Multiple for learning rate decay
  step_size: 4                                  # Apply gamma every step_size
  class_weights: [1.0, 2.0]                     # Weights to apply to each class. A value > 1.0 will apply more weights to the learning of the class.


# Classification parameters; used in image_classification.py --------

classification:
  working_folder: /path/to/images/to/classify           # Folder containing all the images to be classified
  model_name: /path/to/weights/file/last_epoch.pth.tar  # File containing pre-trained weights
```  

## images_to_samples.py  

The first phase of the process is to determine sub-images (samples) to be used for training and validation.  Images to be used mut be of the geotiff type.  Sample locations in each image must be stored in a shapefile (see csv file below).  

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

sample:
  prep_csv_file: /path/to/csv/file_name.csv     # Path to CSV file used in preparation.
  samples_dist: 200                             # Distance (in pixel) between samples
  remove_background: True                       # When True, does not write samples containing only "0" values.
  mask_input_image: False                       # When True, mask the input image where there is no reference data.
```

The csv file must contain 4 comma-separated items: 
- input image file (tif)
- reference vector data (shp)
- attribute of the shapefile to use as classes values
- dataset (either of 'trn' for training or 'val' for validation) where the sample will be used  

Each image is a new line in the csv file.  For example:  

```
\path\to\input\image1.tif,\path\to\reference\vector1.shp,attribute,trn
\path\to\input\image2.tif,\path\to\reference\vector2.shp,attribute,val
```

Outputs:
- 2 .hdfs files with input images and reference data, stored as arrays
    - trn_samples.hdfs
    - val_samples.hdfs

Process:
- Read csv file and for each line in the file, do the following:
    - Create a new raster called "label" with the same properties as the input image
    - Convert shp vector information into the "label" raster. The pixel value is determined by the attribute in the csv file
    - Convert both input and label images to arrays
    - Divide images in smaller samples of size and distance specified in the configuration file. Visual representation of this is provided [here](https://medium.com/the-downlinq/broad-area-satellite-imagery-semantic-segmentation-basiss-4a7ea2c8466f)
    - Write samples into the "val" or "trn" hdfs file, depending on the value contained in the csv file
- Samples are then shuffled to avoid bias in the dat.

## train_model.py

The crux of the learing process is in this phase : training.  Samples labeled "trn" as per above are used to train the neural network.  Samples labeled "val" are used to estimate the training error on a set of sub-images not used for training.

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

training:
  output_path: /path/to/output/weights/folder   # Path to folder where files containing weights will be written
  num_trn_samples: 4960                         # Number of samples to use for training (should be a multiple of batch_size)
  num_val_samples: 2208                         # Number of samples to use for validation (should be a multiple of batch_size)
  pretrained: False                             # .pth.tar filename containig pre-trained weights
  batch_size: 32                                # Size of each batch
  num_epochs: 150                               # Number of epochs
  learning_rate: 0.0001                         # Initial learning rate
  weight_decay: 0                               # Value for weight decay (each epoch)
  gamma: 0.9                                    # Multiple for learning rate decay
  step_size: 4                                  # Apply gamma every step_size
  class_weights: [1.0, 2.0]                     # Weights to apply to each class. A value > 1.0 will apply more weights to the learning of the class.
```

Inputs:
- 1 hdfs file with input images and reference data as arrays used for training (prepared with `images_to_samples.py`)
- 1 hdfs file with input images and reference data as arrays used for validation (prepared with `images_to_samples.py`)

Output:
- Trained model weights
    - checkpoint.pth.tar        Corresponding to the training state where the validation loss was the lowest during the training process.
    - las_epoch.pth.tar         Corresponding to the training state after the last epoch.

Process:
- The application loads the UNet model located in unet_pytorch.py
- Using the hyperparameters provided in `config.yaml` , the application will try to minimize the cross entropy loss on the training and validation data
- For every epoch, the application shows the loss, accuracy, recall and f-score for both datasets (trn and val)
- The application also log the accuracy, recall and f-score for each classes of both the datasets

## image_classification.py

The final step in the process if to assign very pixel in the original image a value corresponding to the most probable class.

To launch the program:  
```
python image_classification.py path/to/config/file/config.yaml
``` 

Details on parameters used by this module:  
```yaml
global:
  number_of_bands: 3        # Number of bands in input images
classification:
  working_folder: /path/to/images/to/classify           # Folder containing all the images to be classified
  model_name: /path/to/weights/file/last_epoch.pth.tar  # File containing pre-trained weights
```
Process:  
- The process will load trained weights to the UNet architecture and perform a per-pixel classification task on all the images contained in the working_folder
