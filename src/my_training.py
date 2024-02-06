import torch
import numpy as np
import lightning.pytorch as pl
import lightning.pytorch.utilities as pl_utils
import segmentation_models_pytorch as smp

import my_utils


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, architecture_config, optimizer_config):
        super().__init__()
        self.autoencoder = smp.create_model(**architecture_config)
        self.optimizer_config = optimizer_config
        self.save_hyperparameters()


    def training_step(self, batch, batch_idx):
        target: torch.Tensor = batch["target"]
        corrupted: torch.Tensor = batch["corrupted"]
        pred: torch.Tensor = self.autoencoder(corrupted)

        loss: torch.Tensor = torch.nn.functional.mse_loss(pred, target)
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
    

    def test_step(self, batch, batch_idx):
        target: torch.Tensor = batch["target"]
        corrupted: torch.Tensor = batch["corrupted"]
        text: list[str] = batch["text"]
        pred: torch.Tensor = self.autoencoder(corrupted)

        metrics = my_utils.compute_metrics(target, pred, text, prefix="test/")
        self.log_dict(metrics, batch_size=target.shape[0])

        return target, corrupted, pred
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_config)
        return optimizer
    

    def on_before_optimizer_step(self, optimizer):
        # Compute the l2-norm for each layer every log step
        if self.global_step % self.trainer.log_every_n_steps == 0:
            norm_type = 2.0
            norms = pl_utils.grad_norm(self.autoencoder, norm_type)
            total_norm = norms[f"grad_{norm_type}_norm_total"]
            self.log("train/total_grad_norm", total_norm)
    

    def predict_step(self, batch, batch_idx):
        return self.autoencoder(batch["image"])
    

    def forward(self, input):
        return self.autoencoder(input)
    
    