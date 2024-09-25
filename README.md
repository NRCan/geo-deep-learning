
## **GDL Refactoring: What was that all about and next steps?**
**Refactoring GDL with PyTorch Lightning**
![lightning Overview](https://github.com/valhassan/geo-deep-learning/blob/573-feature-refactor-geo-deep-learning/images/lightning_overview.png?raw=true)



Since 2020:

![lightning Issues](https://github.com/valhassan/geo-deep-learning/blob/422808634c0446a10efd4bb93ccb506c62deeb30/images/lightning_issues_on_github.png?raw=true)

**Why did we decide to refactor?**

- Access to bleeding edge features.

![lightning Features](https://github.com/valhassan/geo-deep-learning/blob/422808634c0446a10efd4bb93ccb506c62deeb30/images/lightning_features.png?raw=true)
- Less Brittle Code.
- Maintainability.
- Upgrade to industry standards. TorchGeo is built from the ground up with Lightning. 

**4 years has seen consolidation and maturity in the Pytorch Ecosystem.**
<div align="left">
      <a href="https://www.youtube.com/watch?v=5yLzZikS15k">
         <img src="https://i.ytimg.com/vi/rgP_LBtaUEc/maxresdefault.jpg" style="width:100%;">
      </a>
</div>

## **Building Blocks**

### Datasets

<table>
<tr>
<th>GDL-Pytorch</th>
<th>GDL-Lightning</th>
</tr>
<tr>
<td>

```python
class SegmentationDataset(Dataset):
    """Semantic segmentation dataset based on input csvs listing pairs of imagery and ground truth patches as .tif.
    """
```
</td>
<td>

```python
class BlueSkyNonGeo(NonGeoDataset):
    """ 
    This dataset class is intended to handle data for semantic segmentation of geospatial imagery (Binary | Multiclass).
    It loads geospatial image and label patches from csv files.
    """
```

</td>
</tr>
</table>

### Datamodules

<table>
<tr>
<th>GDL-Pytorch</th>
<th>GDL-Lightning</th>
</tr>
<tr>
<td>

```python
def create_dataloader(patches_folder: Path,
                      batch_size: int,
                      gpu_devices_dict: dict,
                      sample_size: int,
                      dontcare_val: int,
                      crop_size: int,
                      num_bands: int,
                      min_annot_perc: int,
                      attr_vals: Sequence,
                      scale: Sequence,
                      cfg: dict,
                      eval_batch_size: int = None,
                      dontcare2backgr: bool = False,
                      compute_sampler_weights: bool = False,
                      debug: bool = False):
    """
    Function to create dataloader objects for training, validation and test datasets.
    @param patches_folder: path to folder containting patches
    @param batch_size: (int) batch size
    @param gpu_devices_dict: (dict) dictionary where each key contains an available GPU with its ram info stored as value
    @param sample_size: (int) size of patches (used to evaluate eval batch-size)
    @param dontcare_val: (int) value in label to be ignored during loss calculation
    @param crop_size: (int) size of one side of the square crop performed on original patch during training
    @param num_bands: (int) number of bands in imagery
    @param min_annot_perc: (int) minimum proportion of ground truth containing non-background information
    @param attr_vals: (Sequence)
    @param scale: (List) imagery data will be scaled to this min and max value (ex.: 0 to 1)
    @param cfg: (dict) Parameters found in the yaml config file.
    @param eval_batch_size: (int) Batch size for evaluation (val and test). Optional, calculated automatically if omitted
    @param dontcare2backgr: (bool) if True, all dontcare values in label will be replaced with 0 (background value)
                            before training
    @param compute_sampler_weights: (bool)
        if True, weights will be computed from dataset patches to oversample the minority class(es) and undersample
        the majority class(es) during training.
    :return: trn_dataloader, val_dataloader, tst_dataloader
    """
    if not patches_folder.is_dir():
        raise FileNotFoundError(f'Could not locate: {patches_folder}')
    experiment_name = patches_folder.stem
    if not len([f for f in patches_folder.glob('*.csv')]) >= 1:
        raise FileNotFoundError(f"Couldn't locate csv file(s) containing list of training data in {patches_folder}")
    num_patches, patches_weight = get_num_patches(patches_path=patches_folder,
                                                  params=cfg,
                                                  min_annot_perc=min_annot_perc,
                                                  attr_vals=attr_vals,
                                                  experiment_name=experiment_name,
                                                  compute_sampler_weights=compute_sampler_weights)
    if not num_patches['trn'] >= batch_size and num_patches['val'] >= batch_size:
        raise ValueError(f"Number of patches is smaller than batch size")
    logging.info(f"Number of patches : {num_patches}\n")
    dataset_constr = create_dataset.SegmentationDataset
    datasets = []

    for subset in ["trn", "val", "tst"]:
        # TODO: should user point to the paths of these csvs directly?
        dataset_file, _ = Tiler.make_dataset_file_name(experiment_name, min_annot_perc, subset, attr_vals)
        dataset_filepath = patches_folder / dataset_file
        datasets.append(dataset_constr(dataset_filepath, num_bands,
                                       max_sample_count=num_patches[subset],
                                       radiom_transform=aug.compose_transforms(params=cfg,
                                                                               dataset=subset,
                                                                               aug_type='radiometric'),
                                       geom_transform=aug.compose_transforms(params=cfg,
                                                                             dataset=subset,
                                                                             aug_type='geometric',
                                                                             dontcare=dontcare_val,
                                                                             crop_size=crop_size),
                                       totensor_transform=aug.compose_transforms(params=cfg,
                                                                                 dataset=subset,
                                                                                 scale=scale,
                                                                                 dontcare2backgr=dontcare2backgr,
                                                                                 dontcare=dontcare_val,
                                                                                 aug_type='totensor'),
                                       debug=debug))
    trn_dataset, val_dataset, tst_dataset = datasets

    # Number of workers
    if cfg.training.num_workers:
        num_workers = cfg.training.num_workers
    else:  # https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/5
        num_workers = len(gpu_devices_dict.keys()) * 4 if len(gpu_devices_dict.keys()) > 1 else 4

    patches_weight = torch.from_numpy(patches_weight)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(patches_weight.type('torch.DoubleTensor'),
                                                             len(patches_weight))

    if gpu_devices_dict and not eval_batch_size:
        max_pix_per_mb_gpu = 280
        eval_batch_size = calc_eval_batchsize(gpu_devices_dict, batch_size, sample_size, max_pix_per_mb_gpu)
    elif not eval_batch_size:
        eval_batch_size = batch_size

    trn_dataloader = DataLoader(trn_dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler,
                                drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=eval_batch_size, num_workers=num_workers, shuffle=False,
                                drop_last=True)
    tst_dataloader = DataLoader(tst_dataset, batch_size=eval_batch_size, num_workers=num_workers, shuffle=False,
                                drop_last=True) if num_patches['tst'] > 0 else None

    if len(trn_dataloader) == 0 or len(val_dataloader) == 0:
        raise ValueError(f"\nTrain and validation dataloader should contain at least one data item."
                         f"\nTrain dataloader's length: {len(trn_dataloader)}"
                         f"\nVal dataloader's length: {len(val_dataloader)}")

    return trn_dataloader, val_dataloader, tst_dataloader
```
This function is housed in train_segmentation.py and calls multiple utility scripts to process data.
Rigid Implementation, near impossible to adapt to a new use case.
</td>
<td>

```python
class BlueSkyNonGeoDataModule(LightningDataModule):
    def __init__(self, 
                 batch_size: int = 16,
                 num_workers: int = 8,
                 data_type_max: int = 255,
                 patch_size: Tuple[int, int] = (512, 512),
                 mean: List[float] = [0.0, 0.0, 0.0],
                 std: List[float] = [1.0, 1.0, 1.0],
                 **kwargs: Any):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.data_type_max = data_type_max
        self.mean = mean
        self.std = std
        self.kwargs = kwargs
        
        self.normalize = K.augmentation.Normalize(mean=self.mean, std=self.std, p=1, keepdim=True)
        random_resized_crop_zoom_in = K.augmentation.RandomResizedCrop(size=self.patch_size, scale=(1.0, 2.0), 
                                                                       p=0.5, keepdim=True)
        random_resized_crop_zoom_out = K.augmentation.RandomResizedCrop(size=self.patch_size, scale=(0.5, 1.0), 
                                                                        p=0.5, keepdim=True)
        
        
        self.transform = AugmentationSequential(K.augmentation.container.ImageSequential
                                                      (K.augmentation.RandomHorizontalFlip(p=0.5, 
                                                                                           keepdim=True),
                                                       K.augmentation.RandomVerticalFlip(p=0.5, 
                                                                                         keepdim=True),
                                                       K.augmentation.RandomAffine(degrees=[-45., 45.], 
                                                                                   p=0.5, 
                                                                                   keepdim=True),
                                                       random_resized_crop_zoom_in,
                                                       random_resized_crop_zoom_out,
                                                       random_apply=1), data_keys=None
                                                      )
    
    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample["image"] /= self.data_type_max
        sample["image"] = self.normalize(sample["image"])
        
        return sample

    def prepare_data(self):
        # download, enhance, tile, etc...
        pass

    def setup(self, stage=None):
        # build the dataset
        train_transform = Compose([self.transform, self.preprocess])
        test_transform = Compose([self.preprocess])
        
        self.train_dataset = BlueSkyNonGeo(split="trn", transforms=train_transform, **self.kwargs)
        self.val_dataset = BlueSkyNonGeo(split="val", transforms=test_transform, **self.kwargs)
        self.test_dataset = BlueSkyNonGeo(split="tst", transforms=test_transform, **self.kwargs)

    def train_dataloader(self) -> DataLoader[Any]:
        
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, shuffle=True)
       
    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False)
```
Single Reusable, Shareable class that encapsulates all the steps needed to process data. 
Flexible Implementation, Anyone can define their own Datamodule as required. 
</td>
</tr>
</table>



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
