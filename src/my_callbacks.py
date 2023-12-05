from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT

import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
import wandb

import my_training
import my_utils


class MyPrintingCallback(Callback):
    def __init__(self, batch_image_idxs: list[int]):
        super().__init__()
        self.batch_image_idxs = batch_image_idxs

    def on_validation_batch_end(self,
        trainer: pl.Trainer,
        pl_module: my_training.LitAutoEncoder,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when the validation batch ends."""

        for batch_i, image_i in self.batch_image_idxs:
            if batch_idx == batch_i:

                target, corrupted, pred = outputs

                # "convert" grayscale to rgb by copying the values to other channels
                target = target.repeat(1, 3, 1, 1)
                pred = pred.repeat(1, 3, 1, 1)

                target_array = my_utils.tensor_to_numpy_for_image(target)
                corrupted_array = my_utils.tensor_to_numpy_for_image(corrupted)
                pred_array = my_utils.tensor_to_numpy_for_image(pred)

                images = np.concatenate([target_array[image_i], corrupted_array[image_i], pred_array[image_i]], axis=1)
                wandb.log({
                    f"images/{batch_idx}_{image_i}_concatenated_img": wandb.Image(images, caption='Target,    Corrupted,    Predicted')
                })

