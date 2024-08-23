import pytorch_lightning as pl
from models.segformer import SegFormer

class SegmentationSegformer(pl.LightningModule):
    def __init__(self, 
                 encoder: str,
                 in_channels: int, 
                 num_classes: int,
                 loss: str = 'cross_entropy'):
        super().__init__()
        self.save_hyperparameters()
        self.model = SegFormer(encoder, in_channels)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        return optimizer

    def train_dataloader(self):
        return DataLoader(train_dataset, batch_size=32, num_workers=4, shuffle=True)

    def val_dataloader(self):
        return DataLoader(val_dataset, batch_size=32, num_workers=4)