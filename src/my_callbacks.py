from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT

import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
import wandb

import my_training
import my_utils


class MyPrintingCallback(Callback):
    def __init__(self, list_idxs: list[int]):
        super().__init__()
        self.list_idxs = list_idxs

    def on_validation_batch_end(self,
        trainer: pl.Trainer,
        pl_module: my_training.LitAutoEncoder,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when the validation batch ends."""

        if batch_idx in self.list_idxs:

            target, corrupted, pred = outputs

            target_array = my_utils.tensor_to_numpy_for_image(target)
            corrupted_array = my_utils.tensor_to_numpy_for_image(corrupted)
            pred_array = my_utils.tensor_to_numpy_for_image(pred)

            images = np.concatenate([target_array[0], corrupted_array[0], pred_array[0]], axis=1)
            wandb.log({
                f"images/{batch_idx}_0_concatenated_img": wandb.Image(images, caption='Target,    Corrupted,    Predicted')
            })

