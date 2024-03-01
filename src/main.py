import os
import yaml
import torch
import torch.utils.data
import torchvision.transforms.functional
import pandas as pd
import lightning.pytorch as pl
from pytorch_lightning.loggers import WandbLogger
import datasets
import tqdm

import my_datamatrix_provider
import my_datasets 
import my_training
import my_callbacks
import my_utils

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

dataloader_train = torch.utils.data.DataLoader(
    dataset=my_datasets.MyIterableDataset(my_datamatrix_provider.DataMatrixProvider()),
    batch_size=config["train_batch_size"]
)

synthetic_dataset = my_datasets.MyMapDatasetFromFolder(folder="./datasets/synthetic_valid_dataset_2")
synthetic_dataloader_valid = torch.utils.data.DataLoader(
    dataset=synthetic_dataset,
    batch_size=config["valid_batch_size"]
)
fst_synthetic_batch = next(iter(synthetic_dataloader_valid)) 
print("synthetic min max:", fst_synthetic_batch["corrupted"].min(), fst_synthetic_batch["corrupted"].max())

real_dataset = my_datasets.MyMapDatasetFromHuggingFace(hf_dataset=datasets.load_dataset("shortery/cropped-dm-codes")["test"])
real_dataloader_valid = torch.utils.data.DataLoader(
    dataset=real_dataset,
    batch_size=config["valid_batch_size"]
)
fst_real_batch = next(iter(real_dataloader_valid)) 
print("real min max:", fst_real_batch["image"].min(), fst_real_batch["image"].max())

os.makedirs("wandb", exist_ok=True)
wandb_logger = WandbLogger(project="dp-dm-codes", save_dir="wandb")
wandb_logger.experiment.config.update(config)
wandb_logger.experiment.define_metric("synthetic_valid/correctly_decoded", summary="max")
wandb_logger.experiment.define_metric("synthetic_valid/decodable", summary="max")
wandb_logger.experiment.define_metric("synthetic_valid/mse_loss", summary="min")
wandb_logger.experiment.define_metric("real_valid/correctly_decoded", summary="max")
wandb_logger.experiment.define_metric("real_valid/decodable", summary="max")

# delete from my local files such "runs" that are already logged to wandb (and older than 24 hours):
# in terminal: wandb sync --cleanndarray


# metrics if the network works perfectly (prediction = target)
perfect_metrics = []
for batch in tqdm.tqdm(synthetic_dataloader_valid, desc="computing metrics if prediction=target"):
    metrics = my_utils.compute_metrics(target=batch["target"], pred=batch["target"], text=batch["text"], prefix="perfect_network/")
    perfect_metrics.append(metrics)
perfect_metrics = pd.DataFrame(perfect_metrics).mean().to_dict()
print("if prediction=target:")
print(perfect_metrics)

# metrics if the network just copies input
# i.e., how many inputs can be (correctly) decoded just by the pylibdmtx code reader
baseline_metrics = []
for batch in tqdm.tqdm(synthetic_dataloader_valid, desc="computing metrics if prediction=input"):
    corrupt = torchvision.transforms.functional.rgb_to_grayscale(batch["corrupted"])
    metrics = my_utils.compute_metrics(target=batch["target"], pred=corrupt, text=batch["text"], prefix="copy_baseline/")
    baseline_metrics.append(metrics)
baseline_metrics = pd.DataFrame(baseline_metrics).mean().to_dict()
print("if prediction=input:")
print(baseline_metrics)
# now for real dataset
real_baseline_metrics = []
for batch in tqdm.tqdm(real_dataloader_valid, desc="computing metrics if prediction=input in real dataset"):
    image = torchvision.transforms.functional.rgb_to_grayscale(batch["image"])
    real_metrics = my_utils.compute_metrics(pred=image, text=batch["text"], prefix="copy_baseline_real/")
    real_baseline_metrics.append(real_metrics)
real_baseline_metrics = pd.DataFrame(real_baseline_metrics).mean().to_dict()
print("if prediction=input in real dataset:")
print(real_baseline_metrics)

autoencoder = my_training.LitAutoEncoder(config["architecture"], config["optimizer"])

os.makedirs("checkpoints", exist_ok=True)
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=f"checkpoints/{wandb_logger.experiment.name}",
    filename="step={step}--corr_dec={real_valid/correctly_decoded:.4f}",
    auto_insert_metric_name=False,
    save_top_k=2,
    save_last=True,
    monitor="real_valid/correctly_decoded",
    mode="max",
)

log_n_predictions = 36
synth_list_idxs = [len(synthetic_dataset)*i//log_n_predictions for i in range(log_n_predictions)]
real_list_idxs = [len(real_dataset)*i//log_n_predictions for i in range(log_n_predictions)]
synth_batch_image_idxs = [(i // config["valid_batch_size"], i % config["valid_batch_size"]) for i in synth_list_idxs]
real_batch_image_idxs = [(i // config["valid_batch_size"], i % config["valid_batch_size"]) for i in real_list_idxs]
batch_image_idxs = [synth_batch_image_idxs, real_batch_image_idxs]
image_callback = my_callbacks.MyPrintingCallback(batch_image_idxs)

early_stop_callback = pl.callbacks.EarlyStopping(
    monitor="synthetic_valid/correctly_decoded",
    mode="max",
    min_delta=0.005,
    patience=5    
)

trainer = pl.Trainer(
    **config["trainer"],
    logger=wandb_logger,
    callbacks=[
        checkpoint_callback,
        image_callback,
        early_stop_callback
    ],
    precision=16
)

wandb_logger.log_metrics(perfect_metrics)
wandb_logger.log_metrics(baseline_metrics)
wandb_logger.log_metrics(real_baseline_metrics)
wandb_logger.log_table(key="validation_characteristics", dataframe=pd.DataFrame([perfect_metrics | baseline_metrics | real_baseline_metrics]))

trainer.fit(
    model=autoencoder,
    train_dataloaders=dataloader_train,
    val_dataloaders=[synthetic_dataloader_valid, real_dataloader_valid],
)
