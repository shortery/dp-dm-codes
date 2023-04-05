import yaml
import torch
import lightning.pytorch as pl
from pytorch_lightning.loggers import WandbLogger

import datamatrix_provider as dmp
import my_datasets 
import my_training

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


wandb_logger = WandbLogger(project="dm-codes", save_dir="checkpoints")

# delete from my local files such "runs" that are already logged to wandb (and older than 24 hours):
# in terminal: wandb sync --clean

autoencoder = my_training.LitAutoEncoder(config["architecture"], config["optimizer"])
my_training = pl.Trainer(**config["trainer"], logger=wandb_logger)
my_training.fit(model=autoencoder, train_dataloaders=dataloader_train)

my_training.logged_metrics
