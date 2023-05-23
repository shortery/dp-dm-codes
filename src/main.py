import os
import yaml
import torch
import lightning.pytorch as pl
from pytorch_lightning.loggers import WandbLogger

import datamatrix_provider as dmp
import my_datasets 
import my_training
import my_callbacks

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
wandb_logger = WandbLogger(project="dm-codes", save_dir="wandb")

# delete from my local files such "runs" that are already logged to wandb (and older than 24 hours):
# in terminal: wandb sync --clean

autoencoder = my_training.LitAutoEncoder(config["architecture"], config["optimizer"])

os.makedirs("checkpoints", exist_ok=True)
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=f"checkpoints/{wandb_logger.experiment.name}",
    save_top_k=2,
    save_last=True,
    monitor="valid/mse_loss",
    mode="min",
)

image_callback = my_callbacks.MyPrintingCallback(num_images_per_batch=1)

trainer = pl.Trainer(
    **config["trainer"],
    logger=wandb_logger,
    callbacks=[checkpoint_callback, image_callback],
)

trainer.fit(
    model=autoencoder,
    train_dataloaders=dataloader_train,
    val_dataloaders=dataloader_valid,
)

trainer.logged_metrics
