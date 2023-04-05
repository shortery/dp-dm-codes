import torch
import lightning.pytorch as pl
import segmentation_models_pytorch as smp


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, architecture_config, optimizer_config):
        super().__init__()
        self.autoencoder = smp.Unet(**architecture_config)
        self.optimizer_config = optimizer_config
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        target = batch["target"]
        corrupted = batch["corrupted"]
        preds = self.autoencoder(corrupted)
        loss = torch.nn.functional.mse_loss(preds, target)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_config)
        return optimizer
    
    