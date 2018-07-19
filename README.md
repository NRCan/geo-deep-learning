# CCMEO's deep learning project

## Requirements  
- Python 3.6 with the following libraries:
    - pytorch 0.4.0
    - torchvision 0.4.0
    - numpy 1.14.0
    - gdal 2.2.2
    - ruamel_yaml 0.15.35
    - scikit-image 0.13.1
    - PIL 5.1.0
    - H5py 2.8.0
- nvidia GPU highly recommended

## config/config.yaml  
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
To launch the program:  
``` 
python images_to_samples.py path/to/config/file/config.yaml
```  
Details on parameters used by this module:
```yaml
global:
  samples_size: 256                 # Size (in pixel) of the samples
  data_path: /path/to/data/folder   # Path to folder containing samples

sample:
  prep_csv_file: /path/to/csv/file_name.csv     # Path to CSV file used in preparation.
  samples_dist: 200                             # Distance (in pixel) between samples
  remove_background: True                       # When True, does not write samples containing only "0" values. 
  mask_input_image: False                       # When True, mask the input image where there is no reference data.
```

The csv file must contain 4 informations, separated by comma:
- input tif
- reference vector data
- attribute to use as value of classes
- dataset ('trn' or 'val') where the image will be used  
Each images is a new line in the csv file.  

``` 
\path\to\input\images.tif,\path\to\reference\vector.shp,attribute,trn
\path\to\input\images2.tif,\path\to\reference\vector.shp,attribute,val
``` 

Outputs:
- 2 .hdfs files with input images and reference data, stored as arrays
    - trn_samples.hdfs
    - val_samples.hdfs

Process: 
- Read csv file and for each line in the file, does the following:
    - Create a new raster called "label" with the same properties as the input image
    - Convert shp vector information into the "label" raster. The pixel value is determined by the attribute in the csv file.
    - Convert both input and label images to arrays
    - Divide images in smaller samples of size and distance specified in the config file. Visual representation of this is provided [here](https://medium.com/the-downlinq/broad-area-satellite-imagery-semantic-segmentation-basiss-4a7ea2c8466f)
    - Write samples into the "val" or "trn" hdfs file, depending on the value contained in the csv file
- The written samples are then shuffled to avoid bias in the data.

## train_model.py
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
- 1 hdfs file with input images and reference data as arrays used for training (prepared with images_to_samples.py)
- 1 hdfs file with input images and reference data as arrays used for validation (prepared with images_to_samples.py)

Output:
- Trained model weights
    - checkpoint.pth.tar        Corresponding to the training state where the validation loss was the lowest during the training process.
    - las_epoch.pth.tar         Corresponding to the training state after the last epoch.

Process:
- The application loads the UNet model located in unet_pytorch.py
- Using the hyperparameters provided in ```config.yaml ``` , the application will try to minimize the crossEntropy loss on the training data and validation data.
- Every epoch, the application shows the loss, accuracy, recall and f-score for both datasets (trn and val).
- The application also log the accuracy, recall and f-score for each classes of both the datasets. 

## image_classification.py
To launch the program:  
``` 
python image_classification.py path/to/config/file/config.yaml
```  
Details on parameters used by this module:  
```yaml
classification:
  working_folder: /path/to/images/to/classify           # Folder containing all the images to be classified
  model_name: /path/to/weights/file/last_epoch.pth.tar  # File containing pre-trained weights
``` 
Process:  
- The process will load trained weights to the UNet architecture and perform a per-pixel classification task on all the images contained in the working_folder.
