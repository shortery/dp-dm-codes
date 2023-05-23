from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT

import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
import wandb

import my_training


class MyPrintingCallback(Callback):
    def __init__(self, num_images_per_batch: int):
        super().__init__()
        self.num_images_per_batch = num_images_per_batch

    def on_validation_batch_end(self,
        trainer: pl.Trainer,
        pl_module: my_training.LitAutoEncoder,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when the validation batch ends."""

        target_array, corrupted_array, preds_array = outputs

        for i in range(self.num_images_per_batch):
            images = np.concatenate([target_array[i], corrupted_array[i], preds_array[i]], axis=1)
            wandb.log({
                f"images_{batch_idx}_{i}/concatenated_img": wandb.Image(images, caption='Target,    Corrupted,    Predicted')
            })

