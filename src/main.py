import os
import yaml
import torch
import torchvision.transforms.functional
import pandas as pd
import lightning.pytorch as pl
from pytorch_lightning.loggers import WandbLogger

import datamatrix_provider as dmp
import my_datasets 
import my_training
import my_callbacks
import my_utils

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

dataloader_valid = torch.utils.data.DataLoader(
    dataset=my_datasets.MyMapDataset(my_datasets.create_dataset(config["valid_size"], config["valid_seed"])),
    batch_size=config["valid_batch_size"]
)
dataloader_test = torch.utils.data.DataLoader(
    dataset=my_datasets.MyMapDataset(my_datasets.create_dataset(config["test_size"], config["test_seed"])),
    batch_size=config["test_batch_size"]
)

dataloader_train = torch.utils.data.DataLoader(
    dataset=my_datasets.MyIterableDataset(dmp.DataMatrixProvider()),
    batch_size=config["train_batch_size"]
)


os.makedirs("wandb", exist_ok=True)
wandb_logger = WandbLogger(project="dp-dm-codes", save_dir="wandb")

# delete from my local files such "runs" that are already logged to wandb (and older than 24 hours):
# in terminal: wandb sync --cleanndarray


# metrics if the network works perfectly (prediction = target)
perfect_metrics = []
for batch in dataloader_valid:
    metrics = my_utils.compute_metrics(target=batch["target"], pred=batch["target"], text=batch["text"], prefix="perfect_network/")
    perfect_metrics.append(metrics)
perfect_metrics = pd.DataFrame(perfect_metrics).mean().to_dict()

# metrics if the network just copies input
# i.e., how many inputs can be (correctly) decoded just by the pylibdmtx code reader
baseline_metrics = []
for batch in dataloader_valid:
    corrupt = torchvision.transforms.functional.rgb_to_grayscale(batch["corrupted"])
    metrics = my_utils.compute_metrics(target=batch["target"], pred=corrupt, text=batch["text"], prefix="copy_baseline/")
    baseline_metrics.append(metrics)
baseline_metrics = pd.DataFrame(baseline_metrics).mean().to_dict()

autoencoder = my_training.LitAutoEncoder(config["architecture"], config["optimizer"])

os.makedirs("checkpoints", exist_ok=True)
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=f"checkpoints/{wandb_logger.experiment.name}",
    save_top_k=2,
    save_last=True,
    monitor="valid/mse_loss",
    mode="min",
)

log_n_predictions = 12
list_idxs = [config["valid_size"]*i//log_n_predictions for i in range(log_n_predictions)]
batch_image_idxs = [(i // config["valid_batch_size"], i % config["valid_batch_size"]) for i in list_idxs]
image_callback = my_callbacks.MyPrintingCallback(batch_image_idxs)

trainer = pl.Trainer(
    **config["trainer"],
    logger=wandb_logger,
    callbacks=[
        checkpoint_callback,
        image_callback
    ],
)

wandb_logger.log_metrics(perfect_metrics)
wandb_logger.log_metrics(baseline_metrics)
wandb_logger.log_table(key="validation_characteristics", dataframe=pd.DataFrame([perfect_metrics | baseline_metrics]))

trainer.fit(
    model=autoencoder,
    train_dataloaders=dataloader_train,
    val_dataloaders=dataloader_valid,
)

trainer.logged_metrics
