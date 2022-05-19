
![Logo](./docs/img/logo.png)

## **Overview**

The **geo-deep-learning** project stems from an initiative at NRCan's [CCMEO](https://www.nrcan.gc.ca/earth-sciences/geomatics/10776).  Its aim is to allow using Convolutional Neural Networks (CNN) with georeferenced data sets.
The overall learning process comprises three broad stages.

### Data preparation
The data preparation phase (sampling) allows creating sub-images (aka chips or patches) that will be used for either training, validation or testing.
The first phase of the process is to determine sub-images (samples) to be used for training, validation and, optionally, test.
Images to be used must be in a format compatible with rasterio/GDAL (ex.: GeoTiff).
Labels (aka annotations) for each image must be stored as polygons in a Geopandas compatible vector file (ex.: GeoPackage).

### Training, along with validation and testing
The training phase is where the neural network learn to use the data prepared in the previous phase to make all the predictions.
The crux of the learning process is the training phase.  

- Samples labeled "*trn*" as per above are used to train the neural network.
- Samples labeled "*val*" are used to estimate the training error (i.e. loss) on a set of sub-images not used for training, after every epoch.
- At the end of all epochs, the model with the lowest error on validation data is loaded and samples labeled "*tst*", if they exist, are used to estimate the accuracy of the model on sub-images unseen during training or validation.

### Inference
The inference phase allows the use of a trained model to predict on new input data.
The final step in the process is to assign every pixel in the original image a value corresponding to the most probable class.

## **Requirement**
This project comprises a set of commands to be run at a shell command prompt.  Examples used here are for a bash shell in an Ubuntu GNU/Linux environment.

- [Python 3.9](https://www.python.org/downloads/release/python-390/), see the full list of dependencies in [environment.yml](environment.yml)
- [hydra](https://hydra.cc/docs/intro/)
- [mlflow](https://mlflow.org/)
- [miniconda](https://docs.conda.io/en/latest/miniconda.html) (highly recommended)
- nvidia GPU (highly recommended)

> The system can be used on your workstation or cluster.

## **Installation**
Those steps are for your workstation on Ubuntu 18.04 using miniconda.
Set and activate your python environment with the following commands:  
```shell
conda env create -f environment.yml
conda activate geo_deep_env
```
> For Windows OS:
> - Install rasterio, fiona and gdal first, before installing the rest. We've experienced some [installation issues](https://github.com/conda-forge/gdal-feedstock/issues/213), with those libraries.
> - Mlflow should be installed using pip rather than conda, as mentioned [here](https://github.com/mlflow/mlflow/issues/1951)

## **Running GDL**
This is an example of how to run GDL with hydra in simple steps with the _**massachusetts buildings**_ dataset in the `/data` folder, for segmentation on buildings: 

1. Clone this github repo.
```shell
git clone https://github.com/NRCan/geo-deep-learning.git
cd geo-deep-learning
```

2. Run the wanted script (for segmentation).
```shell
# Creating the hdf5 from the raw data
python GDL.py mode=sampling
# Training the neural network
python GDL.py mode=train
# Inference on the data
python GDL.py mode=inference
```

> This example is running with the default configuration `./config/gdl_config_template.yaml`, for further examples on running options see the [documentation](config/#Examples).
> You will also find information on how to change the model or add a new one to GDL.

> If you want to introduce a new task like object detection, you only need to add the code in the main folder and name it `object_detection_sampling.py` for example.
> The principle is to name the code like `task_mode.py` and the `GDL.py` will deal with the rest. 
> To run it, you will need to add a new parameter in the command line `python GDL.py mode=sampling task=object_detection` or change the parameter inside the `./config/gdl_config_template.yaml`.

## **Folder Structure**
We suggest the following high level structure to organize the images and the code.
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
├── geo-deep-learning
    └── {scripts as cloned from github}
```
_**Don't forget to change the path of the dataset in the config yaml.**_

> Note: For more information on a subject, go to the specific directory, a `README.md` is provided with all the information and the explanations related to the code.
---

[comment]: <> (## **Segmentation on RGB-NIR images with transfer learning**)

[comment]: <> (![img_rgb_nir]&#40;docs/img/rgb_nir.png&#41;)

[comment]: <> (This section present a different way to use a model with RGB-Nir images. For more informations on the implementation, see the article [Transfer Learning from RGB to Multi-band Imagery]&#40;https://www.azavea.com/blog/2019/08/30/transfer-learning-from-rgb-to-multi-band-imagery/&#41; frome [Azavea]&#40;https://www.azavea.com/&#41;.)

[comment]: <> (Specifications on this functionality:)

[comment]: <> (- At the moment this functionality is only available for the [Deeplabv3 &#40;backbone: resnet101&#41;]&#40;https://arxiv.org/abs/1706.05587&#41;)

[comment]: <> (- You may need to reduce the size of the `batch_size` to fit everything in the memory.)

[comment]: <> (To use this functionality, you will need to change the `global` section of your `yaml` file. The parameters to use this module are:)

[comment]: <> (```yaml)

[comment]: <> (# Global parameters)

[comment]: <> (global:)

[comment]: <> (  samples_size: 256)

[comment]: <> (  num_classes: 4  )

[comment]: <> (  data_path: /home/cauthier/data/)

[comment]: <> (  number_of_bands: 4               # <-- must be 4 for the R-G-B-NIR)

[comment]: <> (  model_name: deeplabv3_resnet101  # <-- must be deeplabv3_resnet101)

[comment]: <> (  task: segmentation               # <-- must be a segmentation task)

[comment]: <> (  num_gpus: 2)

[comment]: <> (  # Module to include the NIR)

[comment]: <> (  modalities: RGBN                 # <-- must be add)

[comment]: <> (  concatenate_depth: 'layer4'      # <-- must specify the point where the NIR will be add)

[comment]: <> (```)

[comment]: <> (The rest of the `yaml` don't have to change.The major changes are the `modalities`, `number_of_bands` and `concatenate_depth` parameters.)

[comment]: <> (If the model select is not `model_name: deeplabv3_resnet101`, but the `number_of_band = 4` and the `modalities = RGBN`, the model will train with the chosen architecture with a input image of 4 dimensions.)

[comment]: <> (Since we have the concatenation point for the **NIR** band only for the `deeplabv3_resnet101`, the `concatenate_depth` parameter option are layers in the `resnet101` backbone: 'conv1', 'maxpool', 'layer2', 'layer3' and 'layer4'.)

[comment]: <> (**Illustration of the principle will fellow soon**)
