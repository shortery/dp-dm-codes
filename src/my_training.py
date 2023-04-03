import torch
import lightning.pytorch as pl

class LitAutoEncoder(pl.LightningModule):
    def __init__(self, autoencoder, learning_rate):
        super().__init__()
        self.autoencoder = autoencoder
        self.learning_rate = learning_rate
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        target = batch["target"]
        corrupted = batch["corrupted"]
        preds = self.autoencoder(corrupted)
        loss = torch.nn.functional.mse_loss(preds, target)
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    