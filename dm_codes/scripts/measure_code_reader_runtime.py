import numpy as np
import pandas as pd
from PIL import Image
import datasets
import lightning.pytorch as pl
import torch
import itertools
from tqdm import tqdm
import random
import typer
import json
import statistics
import timeit

import dm_codes.utils
import dm_codes.training
import dm_codes.datasets

app = typer.Typer(pretty_exceptions_show_locals=False, pretty_exceptions_enable=False)

def prepare_test_dataset(example: dict, random_generator:random.Random) -> dict:
    image: Image.Image = example["image"]
    image = image.resize((128, 128), resample=Image.Resampling.NEAREST)
    angle = random_generator.randint(0, 3) * 90
    image = image.rotate(angle)
    return {"image": image}

def preprocess_image(batch):
    return {"image": dm_codes.datasets._preprocess(np.asarray(batch["image"]))}

@app.command()
def main(
    checkpoint_path: str,
    saved_times_path: str,
    seed: int = 0,
) -> None:
    
    # load dataset from hugging face
    hf_test_dataset = datasets.load_dataset("shortery/dm-codes")["test"]

    # initiate random generator
    random_generator = random.Random(seed)

    # crop and preprocess the image
    hf_dataset = hf_test_dataset.map(dm_codes.datasets.crop_dm_code)
    hf_dataset = hf_dataset.map(lambda x: prepare_test_dataset(x, random_generator))

    # create pandas dataframe and add new collumn decoded_text
    pd_dataset = pd.DataFrame(hf_dataset)
    np_grayscale_images = pd_dataset["image"].map(lambda x: np.asarray(x.convert("L")))
    
    # measure decoding time before nn
    pd_dataset["decoded_text_before_nn"] = pd.Series([None] * len(pd_dataset), dtype=str)
    pd_dataset["decoding_time_before_nn"] = np.zeros(len(pd_dataset))
    for i, img in enumerate(tqdm(np_grayscale_images, desc="Decoding Before NN")):
        start_wall_time = timeit.default_timer()
        decoded_text_before_nn = dm_codes.utils.decode_dm_code(img)
        end_wall_time = timeit.default_timer()
        pd_dataset.at[i, "decoded_text_before_nn"] = decoded_text_before_nn
        pd_dataset.at[i, "decoding_time_before_nn"] = round(end_wall_time - start_wall_time, 4)

    # prepare dataset to be an input to the network
    preprocessed_dataset = hf_dataset.map(preprocess_image)

    # create dataloader
    dataloader_test = torch.utils.data.DataLoader(
        dataset=preprocessed_dataset.with_format("torch"),
        batch_size=64
    )

    # load model from checkpoint and compute predictions
    trainer = pl.Trainer(precision=16)
    loaded_model = dm_codes.training.LitAutoEncoder.load_from_checkpoint(checkpoint_path)
    loaded_model.eval()
    predictions = trainer.predict(loaded_model, dataloader_test)


    # chain all predictions together to get one list
    # (otherwise I could iterate it as "for batch in predictions: for pred in batch: ...")
    pd_dataset["nn_prediction"] = list(itertools.chain(*predictions))

    np_prediction_for_image = pd_dataset["nn_prediction"].map(lambda x: np.squeeze(dm_codes.utils.tensor_to_numpy_for_image(x.unsqueeze(dim=0))))
    pd_dataset["nn_output_image"] = np_prediction_for_image.map(lambda x: Image.fromarray(x, mode="L"))
    
    # measure decoding time after nn
    pd_dataset["decoded_text_after_nn"] = pd.Series([None] * len(pd_dataset), dtype=str)
    pd_dataset["decoding_time_after_nn"] = np.zeros(len(pd_dataset))
    for i, pred in enumerate(tqdm(np_prediction_for_image, desc="Decoding After NN")):
        start_wall_time = timeit.default_timer()
        decoded_text_after_nn = dm_codes.utils.decode_dm_code(pred)
        end_wall_time = timeit.default_timer()
        pd_dataset.at[i, "decoded_text_after_nn"] = decoded_text_after_nn
        pd_dataset.at[i, "decoding_time_after_nn"] = round(end_wall_time - start_wall_time, 4)



    pd_dataset["decoded_before_decoded_after"] = (~pd_dataset["decoded_text_before_nn"].isna()) & (~pd_dataset["decoded_text_after_nn"].isna())
    pd_dataset["decoded_before_not_after"] = (~pd_dataset["decoded_text_before_nn"].isna()) & (pd_dataset["decoded_text_after_nn"].isna())
    pd_dataset["not_before_decoded_after"] = (pd_dataset["decoded_text_before_nn"].isna()) & (~pd_dataset["decoded_text_after_nn"].isna())
    pd_dataset["not_before_not_after"] = (pd_dataset["decoded_text_before_nn"].isna()) & (pd_dataset["decoded_text_after_nn"].isna())

    meadian_decoding_times = {
        # median decoding times for DM codes that were decoded by the code reader both before the NN enhancement and after the NN enhancement
        "decoded_before_decoded_after__before_nn_med_time": pd_dataset[pd_dataset["decoded_before_decoded_after"]]["decoding_time_before_nn"].median(),
        "decoded_before_decoded_after__after_nn_med_time": pd_dataset[pd_dataset["decoded_before_decoded_after"]]["decoding_time_after_nn"].median(),
        # median decoding times for DM codes that were decoded by the code reader before the NN enhancement but not after the NN enhancement
        "decoded_before_not_after__before_nn_med_time": pd_dataset[pd_dataset["decoded_before_not_after"]]["decoding_time_before_nn"].median(),
        "decoded_before_not_after__after_nn_med_time": pd_dataset[pd_dataset["decoded_before_not_after"]]["decoding_time_after_nn"].median(),
        # median decoding times for DM codes that were not decoded by the code reader before the NN enhancement but were decoded after the NN enhancement
        "not_before_decoded_after__before_nn_med_time": pd_dataset[pd_dataset["not_before_decoded_after"]]["decoding_time_before_nn"].median(),
        "not_before_decoded_after__after_nn_med_time": pd_dataset[pd_dataset["not_before_decoded_after"]]["decoding_time_after_nn"].median(),
        # median decoding times for DM codes that were not decoded by the code reader before the NN enhancement nor after the NN enhancement
        "not_before_not_after__before_nn_med_time": pd_dataset[pd_dataset["not_before_not_after"]]["decoding_time_before_nn"].median(),
        "not_before_not_after__after_nn_med_time": pd_dataset[pd_dataset["not_before_not_after"]]["decoding_time_after_nn"].median(),
    } 

    num_examples = {
        "all_examples": len(pd_dataset),
        "num_decoded_before_decoded_after": pd_dataset["decoded_before_decoded_after"].sum().item(),
        "num_decoded_before_not_after": pd_dataset["decoded_before_not_after"].sum().item(),
        "num_not_before_decoded_after": pd_dataset["not_before_decoded_after"].sum().item(),
        "num_not_before_not_after": pd_dataset["not_before_not_after"].sum().item()
    }

    times_dict = {
        "checkpoint_path": checkpoint_path,
        "architecture": loaded_model.hparams["architecture_config"]["arch"],
        "encoder": loaded_model.hparams["architecture_config"]["encoder_name"],
        "meadian_decoding_times_in_seconds": meadian_decoding_times,
        "num_examples": num_examples,
    }


    # "a" means append to file (write just at the end)
    with open(saved_times_path, "a") as file:
        json.dump(times_dict, file)
        file.write("\n")



if __name__ == "__main__":
    app()


# run in terminal:
# $ for ckpt in $(cat checkpoints.txt); do python dm_codes/scripts/measure_code_reader_runtime.py $ckpt "checkpoints_code_reader_runtimes.jsonl"; done
