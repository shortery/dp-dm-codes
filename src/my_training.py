import torch
import numpy as np
import lightning.pytorch as pl
import segmentation_models_pytorch as smp
import pylibdmtx.pylibdmtx

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
        self.log("train/mse_loss", loss)
        return loss


    def validation_step(self, batch, batch_idx):
        target: torch.Tensor = batch["target"]
        corrupted: torch.Tensor = batch["corrupted"]
        preds: torch.Tensor = self.autoencoder(corrupted)

        mse_loss = torch.nn.functional.mse_loss(preds, target)
        self.log("valid/mse_loss", mse_loss)

        decodable = 0
        correctly_decoded = 0
        batch_size, num_channels, w, h = preds.shape
        assert num_channels == 1

        target_array = my_utils.tensor_to_numpy_for_image(target)
        corrupted_array = my_utils.tensor_to_numpy_for_image(corrupted)
        preds_array = my_utils.tensor_to_numpy_for_image(preds)

        correct_pixels = np.mean(target_array == preds_array)

        for i in range(batch_size):
            pred_array = preds_array[i].reshape(w, h)
            decoded_pred = pylibdmtx.pylibdmtx.decode(pred_array)
            if len(decoded_pred) != 0:
                decodable += 1
                if batch["text"][i] == decoded_pred[0].data.decode("utf8"):
                    correctly_decoded += 1
           
        self.log("valid/correct_pixels", correct_pixels)
        self.log("valid/decodable", decodable / batch_size)
        self.log("valid/correctly_decoded", correctly_decoded / batch_size)

        return target_array, corrupted_array, preds_array
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_config)
        return optimizer
    
    