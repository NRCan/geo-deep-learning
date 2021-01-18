
![Logo](./docs/img/logo.png)

## **Overview**

The **geo-deep-learning** project stems from an initiative at NRCan's [CCMEO](https://www.nrcan.gc.ca/earth-sciences/geomatics/10776).  Its aim is to allow using Convolutional Neural Networks (CNN) with georeferenced data sets.
The overall learning process comprises three broad stages:

### Data preparation ([images_to_samples.py](images_to_samples.py))
The data preparation phase (sampling) allows creating sub-images that will be used for either training, validation or testing.

The first phase of the process is to determine sub-images (samples) to be used for training, validation and, optionally, test.  Images to be used must be of the geotiff type.  Sample locations in each image must be stored in a GeoPackage.

> Note: A data analysis module can be found [here](./utils/data_analysis.py) and the documentation in [`docs/README.md`](./docs/README.md). Useful for balancing training data.

### Training, along with validation and testing ([train_segmentation.py](train_segmentation.py))
The training phase is where the neural network learn to use the data prepared in the previous phase to make all the predictions.
The crux of the learning process is the training phase.  

- Samples labeled "*trn*" as per above are used to train the neural network.
- Samples labeled "*val*" are used to estimate the training error (i.e. loss) on a set of sub-images not used for training, after every epoch.
- At the end of all epochs, the model with the lowest error on validation data is loaded and samples labeled "*tst*", if they exist, are used to estimate the accuracy of the model on sub-images unseen during training or validation.

### Inference ([inference.py](inference.py))
The inference phase allows the use of a trained model to predict on new input data.
The final step in the process is to assign every pixel in the original image a value corresponding to the most probable class.

> The training and inference phases currently allow the use of a variety of neural networks to perform classification and semantic segmentation tasks (see the list in [models](models/)).

## **Requirement**
This project comprises a set of commands to be run at a shell command prompt.  Examples used here are for a bash shell in an Ubuntu GNU/Linux environment.

- [Python 3.6](https://www.python.org/downloads/release/python-360/), see the full list of dependencies in [requirements.txt](requirements.txt)
- [mlflow](https://mlflow.org/)
- [minicanda](https://docs.conda.io/en/latest/miniconda.html) (highly recommended)
- nvidia GPU (highly recommended)

> The system can be used on your workstation or cluster and on [AWS](https://aws.amazon.com/).

## **Installation**
Those step are for your a workstation on Ubuntu 18.04 using miniconda.
Set and activate your python environment with the following commands:  
```shell
conda create -n gpu_ENV python=3.6 -c pytorch pytorch torchvision
conda activate gpu_ENV
conda install -c conda-forge ruamel_yaml h5py fiona rasterio geopandas scikit-image scikit-learn tqdm
conda install -c fastai nvidia-ml-py3
conda install mlflow
```
> For Windows OS:
> - Install rasterio, fiona and gdal first, before installing the rest. We've experienced some [installation issues](https://github.com/conda-forge/gdal-feedstock/issues/213), with those libraries.
> - Mlflow should be installed using pip rather than conda, as mentionned [here](https://github.com/mlflow/mlflow/issues/1951)  

## **Folder Structure**
We suggest a high level structure to organize the images and the code.
```
├── {dataset_name}
    └── data
        └── RGB_tiff
            └── {3 band tiff images}
        └── RGBN_tiff
            └── {4 band tiff images}
        └── gpkg
            └── {GeoPackages}
    └── images.csv
    └── yaml_files
            └── your_config.yaml
            ...
            └── different_config.yaml
├── geo-deep-learning
    └── {scripts as cloned from github}
```


## **Running GDL**
1. Clone this github repo.
```shell
git clone https://github.com/NRCan/geo-deep-learning.git
cd geo-deep-learning
```

2. Copy the file `conf/config_template.yaml`, rename it `your_config.yaml` and change the parameters for your needs.
Prepare your data directory and add the paths to a `csv` file.
```shell
# Copying the config_template and rename it at the same time
cp conf/config_template.yaml path/to/yyaml_files/your_config.yaml
# Creating the csv file
touch path/to/images.csv  
```
> See the documentation in the `conf/` directory for more information on how to fill the `yaml` and `csv` files.

3. Execute your task (can be use separately).
```shell
# Creating the hdf5 from the raw data
python images_to_samples.py ./conf/your_config.yaml
# Training the neural network
python train_segmentation.py ./conf/your_config.yaml
# Inference on the new data
python inference.py ./conf/your_config.yaml
```
> If you only want to use the `inference.py` you dont have to fill all the `yaml` file, only fill the inference section.

---

## Mettre des exemples de predictions obtenues sur nos jeux de donn/es


---

## **Segmentation on RGB-NIR images with transfer learning**

![img_rgb_nir](docs/img/rgb_nir.png)

This section present a different way to use a model with RGB-Nir images. For more informations on the implementation, see the article [Transfer Learning from RGB to Multi-band Imagery](https://www.azavea.com/blog/2019/08/30/transfer-learning-from-rgb-to-multi-band-imagery/) frome [Azavea](https://www.azavea.com/).

Specifications on this functionality:
- At the moment this functionality is only available for the [Deeplabv3 (backbone: resnet101)](https://arxiv.org/abs/1706.05587)
- You may need to reduce the size of the `batch_size` to fit everything in the memory.

To use this functionality, you will need to change the `global` section of your `yaml` file. The parameters to use this module are:
```yaml
# Global parameters
global:
  samples_size: 256
  num_classes: 4  
  data_path: /home/cauthier/data/
  number_of_bands: 4               # <-- must be 4 for the R-G-B-NIR
  model_name: deeplabv3_resnet101  # <-- must be deeplabv3_resnet101
  bucket_name:
  task: segmentation               # <-- must be a segmentation task
  num_gpus: 2
  BGR_to_RGB: False                # <-- must be already in RGB
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

  # Module to include the NIR
  modalities: RGBN                 # <-- must be add
  concatenate_depth: 'layer4'      # <-- must specify the point where the NIR will be add
```

The rest of the `yaml` don't have to change.The major changes are the `modalities`, `number_of_bands` and `concatenate_depth` parameters.
If the model select is not `model_name: deeplabv3_resnet101`, but the `number_of_band = 4` and the `modalities = RGBN`, the model will train with the chosen architecture with a input image of 4 dimensions.

Since we have the concatenation point for the **NIR** band only for the `deeplabv3_resnet101`, the `concatenate_depth` parameter option are layers in the `resnet101` backbone: 'conv1', 'maxpool', 'layer2', 'layer3' and 'layer4'.

**Illustration of the principale will fellow soon**



<!-- # Classification Task
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
- The full file path of the classified image, the class identified, as well as the top 5 most likely classes and their value will be displayed on the screen -->
