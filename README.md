
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
      <a href="https://youtu.be/rgP_LBtaUEc?si=4F7P9YbT70EhCqVP">
         <img src="https://i.ytimg.com/vi/rgP_LBtaUEc/maxresdefault.jpg" style="width:100%;">
      </a>
</div>

## **Building Blocks**
![lightning Architecture](https://github.com/valhassan/geo-deep-learning/blob/573-feature-refactor-geo-deep-learning/images/GDL_lightning_architecture.png?raw=true)

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

### LightningModule

<table>
<tr>
<th>GDL-Pytorch</th>
<th>GDL-Lightning</th>
</tr>
<tr>
<td>

```python
def training(train_loader,
          model,
          criterion,
          optimizer,
          scheduler,
          num_classes,
          batch_size,
          ep_idx,
          progress_log,
          device,
          scale,
          vis_params,
          aux_output: bool = False,
          debug=False):
    """
    Train the model and return the metrics of the training epoch

    :param train_loader: training data loader
    :param model: model to train
    :param criterion: loss criterion
    :param optimizer: optimizer to use
    :param scheduler: learning rate scheduler
    :param num_classes: number of classes
    :param batch_size: number of patches to process simultaneously
    :param ep_idx: epoch index (for hypertrainer log)
    :param progress_log: progress log file (for hypertrainer log)
    :param device: device used by pytorch (cpu ou cuda)
    :param scale: Scale to which values in sat img have been redefined. Useful during visualization
    :param vis_params: (Dict) Parameters useful during visualization
    :param debug: (bool) Debug mode
    :return: Updated training loss
    """
    model.train()
    train_metrics = create_metrics_dict(num_classes)

    for batch_index, data in enumerate(tqdm(train_loader, desc=f'Iterating train batches with {device.type}')):
        progress_log.open('a', buffering=1).write(tsv_line(ep_idx, 'trn', batch_index, len(train_loader), time.time()))

        inputs = data["image"].to(device)
        labels = data["mask"].to(device)

        # forward
        optimizer.zero_grad()
        if aux_output:
                outputs, outputs_aux = model(inputs)
        else:
            outputs = model(inputs)
        # added for torchvision models that output an OrderedDict with outputs in 'out' key.
        # More info: https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/
        if isinstance(outputs, OrderedDict):
            outputs = outputs['out']

        # vis_batch_range: range of batches to perform visualization on. see README.md for more info.
        # vis_at_eval: (bool) if True, will perform visualization at eval time, as long as vis_batch_range is valid
        if vis_params['vis_batch_range'] and vis_params['vis_at_train']:
            min_vis_batch, max_vis_batch, increment = vis_params['vis_batch_range']
            if batch_index in range(min_vis_batch, max_vis_batch, increment):
                vis_path = progress_log.parent.joinpath('visualization')
                if ep_idx == 0:
                    logging.info(f'Visualizing on train outputs for batches in range {vis_params["vis_batch_range"]}. '
                                 f'All images will be saved to {vis_path}\n')
                vis_from_batch(vis_params, inputs, outputs,
                               batch_index=batch_index,
                               vis_path=vis_path,
                               labels=labels,
                               dataset='trn',
                               ep_num=ep_idx + 1,
                               scale=scale)
        if aux_output:
            loss_main = criterion(outputs, labels) if num_classes > 1 else criterion(outputs, labels.unsqueeze(1).float())
            loss_aux = criterion(outputs_aux, labels) if num_classes > 1 else criterion(outputs, labels.unsqueeze(1).float())
            loss = 0.4 * loss_aux + loss_main
        else:
            loss = criterion(outputs, labels) if num_classes > 1 else criterion(outputs, labels.unsqueeze(1).float())  
        
        train_metrics['loss'].update(loss.item(), batch_size)

        if device.type == 'cuda' and debug:
            res, mem = gpu_stats(device=device.index)
            logging.debug(OrderedDict(trn_loss=f"{train_metrics['loss'].average():.2f}",
                                      gpu_perc=f"{res['gpu']} %",
                                      gpu_RAM=f"{mem['used'] / (1024 ** 2):.0f}/{mem['total'] / (1024 ** 2):.0f} MiB",
                                      lr=optimizer.param_groups[0]['lr'],
                                      img=data["image"].numpy().shape,
                                      smpl=data["mask"].numpy().shape,
                                      bs=batch_size,
                                      out_vals=np.unique(outputs[0].argmax(dim=0).detach().cpu().numpy()),
                                      gt_vals=np.unique(labels[0].detach().cpu().numpy())))

        loss.backward()
        optimizer.step()

    scheduler.step()
    # if train_metrics["loss"].avg is not None:
    #     logging.info(f'Training Loss: {train_metrics["loss"].avg:.4f}')
    return train_metrics


def evaluation(eval_loader,
               model,
               criterion,
               num_classes,
               batch_size,
               ep_idx,
               progress_log,
               scale,
               vis_params,
               batch_metrics=None,
               dataset='val',
               device=None,
               debug=False
               ):
    """
    Evaluate the model and return the updated metrics
    :param eval_loader: data loader
    :param model: model to evaluate
    :param criterion: loss criterion
    :param num_classes: number of classes
    :param batch_size: number of patches to process simultaneously
    :param ep_idx: epoch index (for hypertrainer log)
    :param progress_log: progress log file (for hypertrainer log)
    :param scale: Scale to which values in sat img have been redefined. Useful during visualization
    :param vis_params: (Dict) Parameters useful during visualization
    :param batch_metrics: (int) Metrics computed every (int) batches. If left blank, will not perform metrics.
    :param dataset: (str) 'val or 'tst'
    :param device: device used by pytorch (cpu ou cuda)
    :param debug: if True, debug functions will be performed
    :return: (dict) eval_metrics
    """
    eval_metrics = create_metrics_dict(num_classes)
    model.eval()

    for batch_index, data in enumerate(tqdm(eval_loader, dynamic_ncols=True, desc=f'Iterating {dataset} '
                                                                                  f'batches with {device.type}')):
        progress_log.open('a', buffering=1).write(tsv_line(ep_idx, dataset, batch_index, len(eval_loader), time.time()))

        with torch.no_grad():
            inputs = data["image"].to(device)
            labels = data["mask"].to(device)
            outputs = model(inputs)

            if isinstance(outputs, OrderedDict):
                outputs = outputs['out']

            # vis_batch_range: range of batches to perform visualization on. see README.md for more info.
            # vis_at_eval: (bool) if True, will perform visualization at eval time, as long as vis_batch_range is valid
            if vis_params['vis_batch_range'] and vis_params['vis_at_eval']:
                min_vis_batch, max_vis_batch, increment = vis_params['vis_batch_range']
                if batch_index in range(min_vis_batch, max_vis_batch, increment):
                    vis_path = progress_log.parent.joinpath('visualization')
                    if ep_idx == 0 and batch_index == min_vis_batch:
                        logging.info(f'\nVisualizing on {dataset} outputs for batches in range '
                                     f'{vis_params["vis_batch_range"]} images will be saved to {vis_path}\n')
                    vis_from_batch(vis_params, inputs, outputs,
                                   batch_index=batch_index,
                                   vis_path=vis_path,
                                   labels=labels,
                                   dataset=dataset,
                                   ep_num=ep_idx + 1,
                                   scale=scale)

            loss = criterion(outputs, labels) if num_classes > 1 else criterion(outputs, labels.unsqueeze(1).float())

            eval_metrics['loss'].update(loss.item(), batch_size)

            if (dataset == 'val') and (batch_metrics is not None):
                # Compute metrics every n batches. Time-consuming.
                if not batch_metrics <= len(eval_loader):
                    logging.error(f"\nBatch_metrics ({batch_metrics}) is smaller than batch size "
                                  f"{len(eval_loader)}. Metrics in validation loop won't be computed")
                if (batch_index + 1) % batch_metrics == 0:  # +1 to skip val loop at very beginning
                    eval_metrics = calculate_batch_metrics(
                        predictions=outputs,
                        gts=labels,
                        n_classes=num_classes,
                        metric_dict=eval_metrics
                    )

            elif dataset == 'tst':
                eval_metrics = calculate_batch_metrics(
                    predictions=outputs,
                    gts=labels,
                    n_classes=num_classes,
                    metric_dict=eval_metrics
                )

            logging.debug(OrderedDict(dataset=dataset, loss=f'{eval_metrics["loss"].avg:.4f}'))

            if debug and device.type == 'cuda':
                res, mem = gpu_stats(device=device.index)
                logging.debug(OrderedDict(
                    device=device, gpu_perc=f"{res['gpu']} %",
                    gpu_RAM=f"{mem['used']/(1024**2):.0f}/{mem['total']/(1024**2):.0f} MiB"
                ))

    if eval_metrics['loss'].average():
        logging.info(f"\n{dataset} Loss: {eval_metrics['loss'].average():.4f}")
    if batch_metrics is not None or dataset == 'tst':
        logging.info(f"\n{dataset} precision: {eval_metrics['precision'].average():.4f}")
        logging.info(f"\n{dataset} recall: {eval_metrics['recall'].average():.4f}")
        logging.info(f"\n{dataset} fscore: {eval_metrics['fscore'].average():.4f}")
        logging.info(f"\n{dataset} iou: {eval_metrics['iou'].average():.4f}")

    return eval_metrics
```
</td>
<td>

```python
class SegmentationSegformer(LightningModule):
    def __init__(self, 
                 encoder: str,
                 in_channels: int, 
                 num_classes: int,
                 loss: Callable,
                 class_labels: List[str] = None,
                 **kwargs: Any):
        super().__init__()
        self.save_hyperparameters(ignore=["loss"])
        self.model = SegFormer(encoder, in_channels, num_classes)
        self.loss = loss
        self.metric= MulticlassJaccardIndex(num_classes=num_classes, average=None, zero_division=np.nan)
        self.labels = [str(i) for i in range(num_classes)] if class_labels is None else class_labels
        self.classwise_metric = ClasswiseWrapper(self.metric, labels=self.labels)

    def forward(self, image: Tensor) -> Tensor:
        return self.model(image)

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        x = batch["image"]
        y = batch["label"]
        y = y.squeeze(1).long()
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        y_hat = y_hat.argmax(dim=1)
        self.log('train_loss', loss, 
                 prog_bar=True, logger=True, 
                 on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["label"]
        y = y.squeeze(1).long()
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        y_hat = y_hat.softmax(dim=1).argmax(dim=1)
        mean_iou = self.metric(y_hat, y)
        self.log('val_loss', loss,
                    prog_bar=True, logger=True, 
                    on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)
        # self.log_dict(val_metrics, 
        #               prog_bar=True, logger=True, 
        #               on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=0)
    
    def test_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["label"]
        y = y.squeeze(1).long()
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        y_hat = y_hat.argmax(dim=1)
        test_metrics = self.classwise_metric(y_hat, y)
        test_metrics["test_loss"] = loss
        self.log_dict(test_metrics, 
                      prog_bar=True, logger=True, 
                      on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)
```

</td>
</tr>
</table>

### Training

<table>
<tr>
<th>GDL-Pytorch</th>
<th>GDL-Lightning</th>
</tr>
<tr>
<td>

```python
logging.info(f'Creating dataloaders from data in {tiling_dir}...\n')
    trn_dataloader, val_dataloader, tst_dataloader = create_dataloader(patches_folder=tiling_dir,
                                                                       batch_size=batch_size,
                                                                       eval_batch_size=eval_batch_size,
                                                                       gpu_devices_dict=gpu_devices_dict,
                                                                       sample_size=patches_size,
                                                                       dontcare_val=dontcare_val,
                                                                       crop_size=crop_size,
                                                                       num_bands=num_bands,
                                                                       min_annot_perc=min_annot_perc,
                                                                       attr_vals=attr_vals,
                                                                       scale=scale,
                                                                       cfg=cfg,
                                                                       dontcare2backgr=dontcare2backgr,
                                                                       compute_sampler_weights=compute_sampler_weights,
                                                                       debug=debug)

    # Save tracking
    set_tracker(mode='train', type='mlflow', task='segmentation', experiment_name=experiment_name, run_name=run_name,
                tracker_uri=tracker_uri, params=cfg,
                keys2log=['general', 'training', 'dataset', 'model', 'optimizer', 'scheduler', 'augmentation'])
    trn_log, val_log, tst_log = [InformationLogger(dataset) for dataset in ['trn', 'val', 'tst']]

    since = time.time()
    best_loss = 999
    last_vis_epoch = 0

    progress_log = output_path / 'progress.log'
    if not progress_log.exists():
        progress_log.open('w', buffering=1).write(tsv_line('ep_idx', 'phase', 'iter', 'i_p_ep', 'time'))  # Add header

    # VISUALIZATION: generate pngs of inputs, labels and outputs
    if vis_batch_range is not None:
        # Make sure user-provided range is a tuple with 3 integers (start, finish, increment).
        # Check once for all visualization tasks.
        if not len(vis_batch_range) == 3 and all(isinstance(x, int) for x in vis_batch_range):
            raise logging.critical(
                ValueError(f'\nVis_batch_range expects three integers in a list: start batch, end batch, increment.'
                           f'Got {vis_batch_range}')
            )
        vis_at_init_dataset = get_key_def('vis_at_init_dataset', cfg['visualization'], 'val')

        # Visualization at initialization. Visualize batch range before first eopch.
        if get_key_def('vis_at_init', cfg['visualization'], False):
            logging.info(f'\nVisualizing initialized model on batch range {vis_batch_range} '
                         f'from {vis_at_init_dataset} dataset...\n')
            vis_from_dataloader(vis_params=vis_params,
                                eval_loader=val_dataloader if vis_at_init_dataset == 'val' else tst_dataloader,
                                model=model,
                                ep_num=0,
                                output_path=output_path,
                                dataset=vis_at_init_dataset,
                                scale=scale,
                                device=device,
                                vis_batch_range=vis_batch_range)

    for epoch in range(0, num_epochs):
        logging.info(f'\nEpoch {epoch}/{num_epochs - 1}\n' + "-" * len(f'Epoch {epoch}/{num_epochs - 1}'))
        # creating trn_report
        trn_report = training(train_loader=trn_dataloader,
                              model=model,
                              criterion=criterion,
                              optimizer=optimizer,
                              scheduler=lr_scheduler,
                              num_classes=num_classes,
                              batch_size=batch_size,
                              ep_idx=epoch,
                              progress_log=progress_log,
                              device=device,
                              scale=scale,
                              vis_params=vis_params,
                              aux_output=aux_output,
                              debug=debug)
        if 'trn_log' in locals():  # only save the value if a tracker is setup
            trn_log.add_values(trn_report, epoch, ignore=['precision', 'recall', 'fscore', 'iou'])
        val_report = evaluation(eval_loader=val_dataloader,
                                model=model,
                                criterion=criterion,
                                num_classes=num_classes,
                                batch_size=batch_size,
                                ep_idx=epoch,
                                progress_log=progress_log,
                                batch_metrics=batch_metrics,
                                dataset='val',
                                device=device,
                                scale=scale,
                                vis_params=vis_params,
                                debug=debug)
        val_loss = val_report['loss'].average()
        if 'val_log' in locals():  # only save the value if a tracker is setup
            if batch_metrics is not None:
                val_log.add_values(val_report, epoch)
            else:
                val_log.add_values(val_report, epoch, ignore=['precision', 'recall', 'fscore', 'iou'])

        if val_loss < best_loss:
            logging.info("\nSave checkpoint with a validation loss of {:.4f}".format(val_loss))  # only allow 4 decimals
            # create the checkpoint file
            checkpoint_tag = checkpoint_stack.pop()
            filename = output_path.joinpath(checkpoint_tag)
            if filename.is_file():
                filename.unlink()
            val_loss_string = f'{val_loss:.2f}'.replace('.', '-')
            checkpoint_tag = f'{experiment_name}_{num_classes}_{"_".join(modalities_str)}_{val_loss_string}.pth.tar'
            filename = output_path.joinpath(checkpoint_tag)
            checkpoint_stack.append(checkpoint_tag)
            best_loss = val_loss
            best_checkpoint_filename = checkpoint_tag
            # More info:
            # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-torch-nn-dataparallel-models
            state_dict = model.module.state_dict() if num_devices > 1 else model.state_dict()
            torch.save({'epoch': epoch,
                        'params': cfg,
                        'model_state_dict': state_dict,
                        'best_loss': best_loss,
                        'optimizer_state_dict': optimizer.state_dict()}, filename)

            # VISUALIZATION: generate pngs of img patches, labels and outputs as alternative to follow training
            if vis_batch_range is not None and vis_at_checkpoint and epoch - last_vis_epoch >= ep_vis_min_thresh:
                if last_vis_epoch == 0:
                    logging.info(f'\nVisualizing with {vis_at_ckpt_dataset} dataset patches on checkpointed model for'
                                 f'batches in range {vis_batch_range}')
                vis_from_dataloader(vis_params=vis_params,
                                    eval_loader=val_dataloader if vis_at_ckpt_dataset == 'val' else tst_dataloader,
                                    model=model,
                                    ep_num=epoch+1,
                                    output_path=output_path,
                                    dataset=vis_at_ckpt_dataset,
                                    scale=scale,
                                    device=device,
                                    vis_batch_range=vis_batch_range)
                last_vis_epoch = epoch

        cur_elapsed = time.time() - since
        # logging.info(f'\nCurrent elapsed time {cur_elapsed // 60:.0f}m {cur_elapsed % 60:.0f}s')
    
    # Script model
    if scriptmodel:
        logging.info(f'\nScripting model...')
        model_to_script = ScriptModel(model,
                                      device=device,
                                      num_classes=num_classes,
                                      input_shape=(1, num_bands, patches_size, patches_size),
                                      mean=mean,
                                      std=std,
                                      scaled_min=scale[0],
                                      scaled_max=scale[1])
        
        scripted_model = torch.jit.script(model_to_script)
        if best_checkpoint_filename is not None:
            scripted_model_filename = best_checkpoint_filename.replace('.pth.tar', '_scripted.pt')
            scripted_model.save(output_path.joinpath(scripted_model_filename))
        else:
            scripted_model_filename = f'{experiment_name}_{num_classes}_{"_".join(modalities_str)}_scripted.pt'
            scripted_model.save(output_path.joinpath(scripted_model_filename))

    # load checkpoint model and evaluate it on test dataset.
    if int(cfg['general']['max_epochs']) > 0:   # if num_epochs is set to 0, model is loaded to evaluate on test set
        checkpoint = read_checkpoint(filename)
        checkpoint = adapt_checkpoint_to_dp_model(checkpoint, model)
        model.load_state_dict(state_dict=checkpoint['model_state_dict'])

    if tst_dataloader:
        tst_report = evaluation(eval_loader=tst_dataloader,
                                model=model,
                                criterion=criterion,
                                num_classes=num_classes,
                                batch_size=batch_size,
                                ep_idx=num_epochs,
                                progress_log=progress_log,
                                batch_metrics=batch_metrics,
                                dataset='tst',
                                scale=scale,
                                vis_params=vis_params,
                                device=device)
        if 'tst_log' in locals():  # only save the value if a tracker is set                                     up
            tst_log.add_values(tst_report, num_epochs)
```
</td>
<td>

```python
class GeoDeepLearningCLI(LightningCLI):
    def before_fit(self):
        self.datamodule.prepare_data()
        self.datamodule.setup("fit")
        self.print_dataset_sizes()
    
    def after_fit(self):
        if self.trainer.is_global_zero:
            best_model_path = self.trainer.checkpoint_callback.best_model_path
            test_trainer = Trainer(devices=1, 
                                   accelerator="auto", 
                                   strategy="auto")
            best_model = self.model.__class__.load_from_checkpoint(best_model_path)
            test_trainer.test(model=best_model, dataloaders=self.datamodule.test_dataloader())
        self.trainer.strategy.barrier()
    
    def print_dataset_sizes(self):
        if self.trainer.is_global_zero:
            train_size = len(self.datamodule.train_dataloader().dataset)
            val_size = len(self.datamodule.val_dataloader().dataset)
            test_size = len(self.datamodule.test_dataloader().dataset)

            print(f"Number of training samples: {train_size}")
            print(f"Number of validation samples: {val_size}")
            print(f"Number of test samples: {test_size}")


def main(args: ArgsType = None) -> None:
    cli = GeoDeepLearningCLI(save_config_kwargs={"overwrite": True}, 
                             args=args)
    if cli.trainer.is_global_zero:
        print("Done!")
    

if __name__ == "__main__":
    main()
```

</td>
</tr>
</table>

### Features

#### Metrics
<table>
<tr>
<th>GDL-Pytorch</th>
<th>GDL-Lightning</th>
</tr>
<tr>
<td>

```python

def calculate_batch_metrics(
        predictions: torch.Tensor,
        gts: torch.Tensor,
        n_classes: int,
        metric_dict: Dict) -> Dict:
    """
    Calculate batch metrics for the given batch and ground-truth labels.
    Update current metrics dictionary.
    Args:
        predictions: predicted logits, the direct outputs from the model.
        gts: ground-truth labels.
        n_classes: number of segmentation classes. If it equals to 1, then converted to a binary problem.
        metric_dict: current metrics dictionary.

    Returns:
        Updated metrics dictionary.
    """
    # Extract the batch size:
    batch_size = predictions.shape[0]

    # Get hard labels from the predicted logits:
    if n_classes == 1:
        label_pred = torch.sigmoid(predictions).clone().cpu().detach().numpy()
        ones_array = np.ones(shape=label_pred.shape, dtype=np.uint8)
        bg_pred = ones_array - label_pred
        label_pred = np.concatenate([bg_pred, label_pred], axis=1)
    elif n_classes > 1:
        label_pred = F.softmax(predictions.clone().cpu().detach(), dim=1).numpy()
    else:
        raise ValueError(f'Number of classes is less than 1 == {n_classes}')

    label_pred = np.array(np.argmax(label_pred, axis=1)).astype(np.uint8)
    label_true = gts.clone().cpu().detach().numpy().copy()

    # Make a problem binary if the number of classes == 1:
    n_classes = 2 if n_classes == 1 else n_classes

    # Initialize an empty batch confusion matrix:
    batch_matrix = np.zeros((n_classes, n_classes))

    # For each sample in the batch, add the confusion matrix the to the batch matrix:
    for lp, lt in zip(label_pred, label_true):
        batch_matrix += calculate_confusion_matrix(lp.flatten(), lt.flatten(), n_classes)

    # Calculate metrics from the confusion matrix:
    iu = np.diag(batch_matrix) / (batch_matrix.sum(axis=1) + batch_matrix.sum(axis=0) - np.diag(batch_matrix))
    precision = np.diag(batch_matrix) / batch_matrix.sum(axis=1)
    recall = np.diag(batch_matrix) / batch_matrix.sum(axis=0)
    f_score = 2 * ((precision * recall) / (precision + recall))

    # Update the metrics dict:
    mean_iu = np.nanmean(iu)
    metric_dict['iou'].update(mean_iu, batch_size)

    mean_iu_nobg = np.nanmean(iu[1:])
    metric_dict['iou_nonbg'].update(mean_iu_nobg, batch_size)

    mean_precision = np.nanmean(precision)
    metric_dict['precision'].update(mean_precision, batch_size)

    mean_recall = np.nanmean(recall)
    metric_dict['recall'].update(mean_recall, batch_size)

    mean_f_score = np.nanmean(f_score)
    metric_dict['fscore'].update(mean_f_score, batch_size)

    cls_list = [str(cls) for cls in range(n_classes)]

    for i, cls_lbl in enumerate(cls_list):
        metric_dict['iou_' + cls_lbl].update(iu[i], batch_size)
        metric_dict['precision_' + cls_lbl].update(precision[i], batch_size)
        metric_dict['recall_' + cls_lbl].update(recall[i], batch_size)
        metric_dict['fscore_' + cls_lbl].update(f_score[i], batch_size)

    return metric_dict
```
</td>
<td>

```python
# Introducing Torch metrics
from torchmetrics.classification import BinaryAccuracy

train_accuracy = BinaryAccuracy()
valid_accuracy = BinaryAccuracy()
```
Multi-GPU support out of the box 

</td>
</tr>
</table>

#### Multi-GPU Support with Different Strategies (DDP, FSDP, DeepSpeed) 
<table>
<tr>
<th>GDL-Pytorch</th>
<th>GDL-Lightning</th>
</tr>
<tr>
<td>

```python
class TrainEngine:
    def __init__(self, multiproc: DictConfig, engine_type: str = 'cpu'):
        super(TrainEngine, self).__init__()
        self.engine_type = engine_type
        self.ddp_initialized = False
        self.multiproc = multiproc
        self.gpu_devices_dict = {}
        self.gpu_ids = list(self.gpu_devices_dict.keys())

        if self.multiproc.gpus and not self.engine_type == 'cpu':
            self.gpu_devices_dict = get_device_ids(self.multiproc.gpus)
            self.gpu_ids = list(self.gpu_devices_dict.keys())

        if self.engine_type == 'distributed_data_parallel':
            # set up distributed data parallel
            if set(self.multiproc.local_env_var).issubset(os.environ):
                self.multiproc.global_rank = int(os.environ["RANK"])
                self.multiproc.local_rank = int(os.environ["LOCAL_RANK"])
                self.multiproc.ntasks = int(os.environ["LOCAL_WORLD_SIZE"])
                self.multiproc.world_size = int(os.environ["WORLD_SIZE"])

                dist.init_process_group(backend=self.multiproc.dist_backend,
                                        init_method=self.multiproc.dist_url,
                                        rank=self.multiproc.global_rank,
                                        world_size=self.multiproc.world_size)

            elif set(self.multiproc.hpc_env_var).issubset(os.environ):
                self.multiproc.global_rank = int(os.environ["SLURM_PROCID"])
                self.multiproc.local_rank = int(os.environ["SLURM_LOCALID"])
                self.multiproc.ntasks = int(os.environ["SLURM_NTASKS_PER_NODE"])
                self.multiproc.world_size = int(os.environ["WORLD_SIZE"])

                dist.init_process_group(backend=self.multiproc.dist_backend,
                                        init_method=self.multiproc.dist_url,
                                        rank=self.multiproc.global_rank,
                                        world_size=self.multiproc.world_size
                                        )
            if dist.is_initialized():
                self.ddp_initialized = True
            else:
                raise TypeError(f"Distributed Data Parallel is not initialized try CPU/DataParallel engine")

    # Set device(s)

    def get_device(self):

        if self.engine_type == "cpu":
            device = self.engine_type
            return device

        if self.engine_type == "data_parallel":
            device = set_device(gpu_devices_dict=self.gpu_devices_dict)
            return device

        if self.engine_type == "distributed_data_parallel":
            device = torch.device(f"cuda:{self.multiproc.local_rank}")
            return device

    def prepare_model(self, model: torch.nn.Module):
        device = self.get_device()
        model.to(device)

        if self.engine_type == 'cpu':
            return model

        if self.engine_type == 'data_parallel':
            model = to_dp_model(model=model, devices=self.gpu_ids) if len(self.gpu_ids) > 1 else model
            return model

        if self.engine_type == "distributed_data_parallel":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                              device_ids=[self.multiproc.local_rank],
                                                              output_device=self.multiproc.local_rank,
                                                              find_unused_parameters=True)
            return model

    def prepare_dataloader(self,
                           datasets: Sequence[Dataset],
                           samples_weight: Sequence[float],
                           num_samples: Dict[str, int],
                           batch_size: int, eval_batch_size: int,
                           sample_size: int, num_workers: int = 0):

        # https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/5
        if self.engine_type == 'data_parallel' and num_workers == 0:
            num_workers = len(self.gpu_ids) * 4 if len(self.gpu_ids) > 1 else 4

        # if self.engine_type == "distributed_data_parallel":
        #     trn_sampler = DistributedSamplerWrapper(trn_sampler)

        if self.gpu_devices_dict and not eval_batch_size:
            max_pix_per_mb_gpu = 280  # TODO: this value may need to be finetuned
            eval_batch_size = calc_eval_batchsize(self.gpu_devices_dict, batch_size, sample_size, max_pix_per_mb_gpu)
        elif not eval_batch_size:
            eval_batch_size = batch_size

        trn_dataset, val_dataset, tst_dataset = datasets
        val_sampler = None
        if self.engine_type == "distributed_data_parallel":
            trn_sampler = DistributedSampler(trn_dataset)
            val_sampler = DistributedSampler(val_dataset)
        else:
            samples_weight = torch.from_numpy(samples_weight)
            trn_sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), 
                                                                         len(samples_weight))

        trn_dataloader = DataLoader(trn_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                    sampler=trn_sampler, drop_last=True)
        val_dataloader = DataLoader(val_dataset, batch_size=eval_batch_size, num_workers=num_workers, pin_memory=True,
                                    sampler=val_sampler, drop_last=True)
        tst_dataloader = DataLoader(tst_dataset, batch_size=eval_batch_size, num_workers=num_workers,
                                    pin_memory=True, shuffle=False, drop_last=True) if num_samples['tst'] > 0 else None

        if len(trn_dataloader) == 0 or len(val_dataloader) == 0:
            raise ValueError(f"\nTrain and validation dataloader should contain at least one data item."
                             f"\nTrain dataloader's length: {len(trn_dataloader)}"
                             f"\nVal dataloader's length: {len(val_dataloader)}")

        return trn_dataloader, val_dataloader, tst_dataloader
```
</td>
<td>

```python
class Trainer:
    def __init__(self,
                 cfg: DictConfig) -> None:
        """ 
        Train and validate a model for semantic segmentation.
        
        :param cfg: (dict) Parameters found in the yaml config file.
        
        """
        self.cfg = cfg
        
        
         # TRAIN ACCELERATOR AND STRATEGY PARAMETERS
        num_devices = get_key_def('num_gpus', cfg['training'], default=0)
        num_nodes = get_key_def('num_nodes', self.cfg['training'], default=1)
        num_tasks = get_key_def('num_tasks', self.cfg['training'], default=0)
        self.strategy = get_key_def("strategy", self.cfg['training'], default="dp")
        precision = get_key_def("precision", self.cfg['training'], default="32-true")
        if num_devices and not num_devices >= 0:
            raise ValueError("\nMissing mandatory num gpus parameter")
        if self.strategy == "dp":
            num_tasks = num_devices
        if self.strategy == "ddp":
            self.strategy = DDPStrategy(find_unused_parameters=True)
        accelerator = get_key_def("accelerator", self.cfg['training'], default="cuda")
        self.fabric = Fabric(accelerator=accelerator, devices=num_tasks, 
                             num_nodes=num_nodes, strategy=self.strategy, precision=precision)
        self.fabric.launch()
```

</td>

<td>

```yaml
trainer:
  accelerator: "gpu"
  devices: -1
  strategy: "ddp"
```

</td>
</tr>
</table>

### Other Features
- logging
- early stopping
- visualization
- checkpointing 
- model scripting

All other features are abstracted away by Lightning. No more tedious Boilerplate code!


## Next Steps

Collective effort is required!

- Refactor branch created; the new code base will live there until we are ready to merge squatch "develop branch"! 
- Project will be created on Github to manage milestones and tasks.
- Documentation, Lightning modules as applicable, Some tests, CI/CD Github workflow, e.t.c are required.
