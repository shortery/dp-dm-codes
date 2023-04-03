import yaml
import torch
import lightning.pytorch as pl
import segmentation_models_pytorch as smp

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

autoencoder = my_training.LitAutoEncoder(smp.Unet(**config["architecture"]), **config["optimizer"])
my_training = pl.Trainer(**config["trainer"])
my_training.fit(model=autoencoder, train_dataloaders=dataloader_train)

my_training.logged_metrics