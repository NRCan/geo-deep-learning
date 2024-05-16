
![Logo](./docs/img/logo.png)

## **About**

The **Geo-Deep-Learning** project stems from an initiative at NRCan's [CCMEO](https://www.nrcan.gc.ca/earth-sciences/geomatics/10776).  Its aim is to allow using Convolutional Neural Networks (CNN) with georeferenced datasets.

In **Geo-Deep-Learning**, the learning process comprises two broad stages: tiling and training, followed by inference, which makes use of a trained model to make new predictions on unseen imagery. 

## **Requirement**
This project comprises a set of commands to be run at a shell command prompt.  Examples used here are for a bash shell in an Ubuntu GNU/Linux environment.

- [Python 3.10](https://www.python.org/downloads/release/python-3100/), see the full list of dependencies in [environment.yml](environment.yml)
- [hydra](https://hydra.cc/docs/intro/)
- [mlflow](https://mlflow.org/)
- [miniconda](https://docs.conda.io/en/latest/miniconda.html) (highly recommended)
- nvidia GPU (highly recommended)

## **Installation**
Miniconda is suggested as the package manager for GDL. However, users are advised to [switch to libmamba](https://github.com/NRCan/geo-deep-learning#quickstart-with-conda) as conda's default solver or to __directly use mamba__ instead of conda if they are facing extended installation time or other issues. Additional problems are grouped in the [troubleshooting section](https://github.com/NRCan/geo-deep-learning#troubleshooting). If issues persist, users are encouraged to open a new issue for assistance.

> Tested on Ubuntu 20.04, Windows 10 and WSL 2.

### Quickstart with conda
To execute scripts in this project, first create and activate your python environment with the following commands:  
```shell
$ conda env create -f environment.yml
$ conda activate geo_deep_env
```

### Change conda's default solver for faster install (__Optional__)
```shell
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

### Troubleshooting
- *ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found*
  - Export path to library or set it permenantly in your .bashrc file (example with conda) :
    ```bash
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
    ```

## **How to use?**
This is an example of how to run GDL with hydra in simple steps with the _**massachusetts buildings**_ dataset in the `tests/data/` folder, for segmentation on buildings: 

1. Clone this github repo.
```shell
(geo_deep_env) $ git clone https://github.com/NRCan/geo-deep-learning.git
(geo_deep_env) $ cd geo-deep-learning
```

2. Run the wanted script (for segmentation).
```shell
# Creating the patches from the raw data
(geo_deep_env) $ python GDL.py mode=tiling
# Training the neural network
(geo_deep_env) $ python GDL.py mode=train
# Inference on the data
(geo_deep_env) $ python GDL.py mode=inference
```

This example runs with a default configuration `./config/gdl_config_template.yaml`. For further examples on configuration options see the configuration documentation.
To see the different mode and task available go see the documentation here.

### New task
If you want to introduce a new task like object detection, you only need to add the code in the main folder and name it `object_detection_tiling.py` for example.
The principle is to name the code like `{task}_{mode}.py` and the `GDL.py` will deal with the rest. 
To run it, you will need to add a new parameter in the command line `python GDL.py mode=tiling task=object_detection` or change the parameter inside the `./config/gdl_config_template.yaml`.

## **Contributing**
We welcome all forms of user contributions including feature requests, bug reports, code, documentation requests, and code. Simply open an issue in the tracker.

If you think you're not skilled or experienced enough to contribute, this is **not TRUE**!
Don't be affraid to help us, every one start somewhere, and it will be our pleasure to help you
to help us. 

You can find more information on how to create a good issue on a GitHub project [Here](https://docs.github.com/en/issues/tracking-your-work-with-issues/creating-an-issue).


After creating an issue, you can start working on the solution. 
When you have finish working on your code, it's time for the **PR**.
All the information on how to create a good **PR** on a GitHub project [Here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request).

## **Citing Geo Deep Learning**
Citations help us justify the effort that goes into building and maintaining this project.
If you used _**Geo Deep Learning**_ for your research, please consider citing us.

```
@misc{NRCAN:2020,
  Author = {Natural Resources Canada, Government of Canada},
  Title = {Geo Deep Learning},
  Year = {2020},
  Publisher = {GitHub},
  Journal = {GitHub repository},
  Howpublished = {\url{https://github.com/NRCan/geo-deep-learning}}
}
```

Or you can also use the [CITATION.cff](https://github.com/NRCan/geo-deep-learning/blob/develop/CITATION.cff) file to cite this project.

## **Contacting us**
The best way to get in touch is to open an issue or comment on any open [issue](https://github.com/NRCan/geo-deep-learning/issues/new) or pull request. 

## **License**
Project is distributed under [MIT License](https://github.com/NRCan/geo-deep-learning/blob/develop/LICENSE).


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
