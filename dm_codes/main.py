import os
import yaml
import torch
import torch.utils.data
import torchvision.transforms.functional
import pandas as pd
import lightning.pytorch as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
import wandb.sdk
import datasets
import tqdm

import dm_codes.my_datamatrix_provider
import dm_codes.my_datasets 
import dm_codes.my_training
import dm_codes.my_callbacks
import dm_codes.my_utils

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

train_dataloader = torch.utils.data.DataLoader(
    dataset=dm_codes.my_datasets.MyIterableDataset(dm_codes.my_datamatrix_provider.DataMatrixProvider()),
    batch_size=config["train_batch_size"]
)

synthetic_valid_dataset = dm_codes.my_datasets.MyMapDatasetFromFolder(folder="./datasets/synthetic_valid_dataset_3")
synthetic_valid_dataloader = torch.utils.data.DataLoader(
    dataset=synthetic_valid_dataset,
    batch_size=config["valid_batch_size"]
)
fst_synthetic_batch = next(iter(synthetic_valid_dataloader)) 
print("synthetic min max:", fst_synthetic_batch["corrupted"].min(), fst_synthetic_batch["corrupted"].max())

hf_valid_dataset = datasets.load_dataset("shortery/dm-codes")["validation"]
real_valid_dataset = dm_codes.my_datasets.MyMapDatasetFromHuggingFace(hf_dataset=hf_valid_dataset.map(dm_codes.my_datasets.crop_dm_code))
real_valid_dataloader = torch.utils.data.DataLoader(
    dataset=real_valid_dataset,
    batch_size=config["valid_batch_size"]
)
fst_real_batch = next(iter(real_valid_dataloader)) 
print("real min max:", fst_real_batch["image"].min(), fst_real_batch["image"].max())

os.makedirs("wandb", exist_ok=True)
wandb_logger = WandbLogger(project="dp-dm-codes", save_dir="wandb")
wandb_experiment = wandb_logger.experiment
assert isinstance(wandb_experiment, wandb.sdk.wandb_run.Run)

wandb_experiment.config.update(config)
wandb_experiment.define_metric("synthetic_valid/correctly_decoded", summary="max")
wandb_experiment.define_metric("synthetic_valid/decodable", summary="max")
wandb_experiment.define_metric("synthetic_valid/mse_loss", summary="min")
wandb_experiment.define_metric("real_valid/correctly_decoded", summary="max")
wandb_experiment.define_metric("real_valid/decodable", summary="max")


# metrics if the network works perfectly (prediction = target)
perfect_metrics = []
for batch in tqdm.tqdm(synthetic_valid_dataloader, desc="computing metrics if prediction=target"):
    metrics = dm_codes.my_utils.compute_metrics(target=batch["target"], pred=batch["target"], text=batch["text"], prefix="perfect_network/")
    perfect_metrics.append(metrics)
perfect_metrics = pd.DataFrame(perfect_metrics).mean().to_dict()
print("if prediction=target:")
print(perfect_metrics)

# metrics if the network just copies input
# i.e., how many inputs can be (correctly) decoded just by the pylibdmtx code reader
baseline_metrics = []
for batch in tqdm.tqdm(synthetic_valid_dataloader, desc="computing metrics if prediction=input"):
    corrupt = torchvision.transforms.functional.rgb_to_grayscale(batch["corrupted"])
    metrics = dm_codes.my_utils.compute_metrics(target=batch["target"], pred=corrupt, text=batch["text"], prefix="copy_baseline/")
    baseline_metrics.append(metrics)
baseline_metrics = pd.DataFrame(baseline_metrics).mean().to_dict()
print("if prediction=input:")
print(baseline_metrics)
# now for real dataset
real_baseline_metrics = []
for batch in tqdm.tqdm(real_valid_dataloader, desc="computing metrics if prediction=input in real dataset"):
    image = torchvision.transforms.functional.rgb_to_grayscale(batch["image"])
    real_metrics = dm_codes.my_utils.compute_metrics(pred=image, text=batch["text"], prefix="copy_baseline_real/")
    real_baseline_metrics.append(real_metrics)
real_baseline_metrics = pd.DataFrame(real_baseline_metrics).mean().to_dict()
print("if prediction=input in real dataset:")
print(real_baseline_metrics)

autoencoder = dm_codes.my_training.LitAutoEncoder(config["architecture"], config["optimizer"])

os.makedirs("checkpoints", exist_ok=True)
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=f"checkpoints/{wandb_experiment.name}",
    filename="step={step}--corr_dec={real_valid/correctly_decoded:.4f}",
    auto_insert_metric_name=False,
    save_top_k=2,
    save_last=True,
    monitor="real_valid/correctly_decoded",
    mode="max",
)

log_n_predictions = 36
synth_list_idxs = [len(synthetic_valid_dataset)*i//log_n_predictions for i in range(log_n_predictions)]
real_list_idxs = [len(real_valid_dataset)*i//log_n_predictions for i in range(log_n_predictions)]
synth_batch_image_idxs = [(i // config["valid_batch_size"], i % config["valid_batch_size"]) for i in synth_list_idxs]
real_batch_image_idxs = [(i // config["valid_batch_size"], i % config["valid_batch_size"]) for i in real_list_idxs]
batch_image_idxs = [synth_batch_image_idxs, real_batch_image_idxs]
image_callback = dm_codes.my_callbacks.MyPrintingCallback(batch_image_idxs)

early_stop_callback = pl.callbacks.EarlyStopping(**config["early_stopping"])

trainer = pl.Trainer(
    **config["trainer"],
    logger=wandb_logger,
    callbacks=[
        checkpoint_callback,
        image_callback,
        early_stop_callback
    ]
)


wandb_experiment.log(perfect_metrics, step=0)
wandb_experiment.log(baseline_metrics, step=0)
wandb_experiment.log(real_baseline_metrics, step=0)
wandb_experiment.log({
    "validation_characteristics":
    wandb.Table(dataframe=pd.DataFrame([perfect_metrics | baseline_metrics | real_baseline_metrics]))
}, step=0)

trainer.validate(model=autoencoder, dataloaders=[synthetic_valid_dataloader, real_valid_dataloader], verbose=False)

trainer.fit(
    model=autoencoder,
    train_dataloaders=train_dataloader,
    val_dataloaders=[synthetic_valid_dataloader, real_valid_dataloader],
)
