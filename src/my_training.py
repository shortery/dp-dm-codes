import torch
import numpy as np
import lightning.pytorch as pl
import segmentation_models_pytorch as smp

import my_utils


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
        self.log("train/mse_loss", loss, batch_size=target.shape[0])
        return loss


    def validation_step(self, batch, batch_idx):
        target: torch.Tensor = batch["target"]
        corrupted: torch.Tensor = batch["corrupted"]
        text: list[str] = batch["text"]
        pred: torch.Tensor = self.autoencoder(corrupted)

        metrics = my_utils.compute_metrics(target, pred, text, prefix="valid/")
        self.log_dict(metrics, batch_size=target.shape[0])

        return target, corrupted, pred
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_config)
        return optimizer
    
    