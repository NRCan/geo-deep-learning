import pytorch_lightning as pl
from typing import Any, Callable
from hydra.utils import instantiate
from models.segformer import SegFormer

class SegmentationSegformer(pl.LightningModule):
    def __init__(self, 
                 encoder: str,
                 in_channels: int, 
                 num_classes: int,
                 loss: Callable, 
                 **kwargs: Any):
        super().__init__()
        self.save_hyperparameters()
        self.model = SegFormer(encoder, in_channels, num_classes)
        self.loss = loss

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = instantiate(self.hparams["optimizer"], params=self.model.parameters())
        scheduler = instantiate(self.hparams["lr_scheduler"])
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss"
                }
        }
    

    def train_dataloader(self):
        return DataLoader(train_dataset, batch_size=32, num_workers=4, shuffle=True)

    def val_dataloader(self):
        return DataLoader(val_dataset, batch_size=32, num_workers=4)