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
    saved_accuracies_path: str,
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
    pd_dataset["decoded_text_before_nn"] = np_grayscale_images.map(lambda x: dm_codes.utils.decode_dm_code(x))
    
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
    pd_dataset["decoded_text_after_nn"] = np_prediction_for_image.map(dm_codes.utils.decode_dm_code)

    accuracies_dict = {
        "checkpoint_path": checkpoint_path,
        "architecture": loaded_model.hparams["architecture_config"]["arch"],
        "encoder": loaded_model.hparams["architecture_config"]["encoder_name"],
    }

    num_all = len(pd_dataset)
    for decoded_text_column in ["decoded_text_before_nn", "decoded_text_after_nn"]:
        num_correctly_decoded = len(pd_dataset[(pd_dataset[decoded_text_column] == pd_dataset["text"])])
        num_decoded = len(pd_dataset[(~pd_dataset[decoded_text_column].isna())])
        if decoded_text_column == "decoded_text_before_nn":
            key_string = "before_nn_"
        elif decoded_text_column == "decoded_text_after_nn":
            key_string = "after_nn_"
        
        accuracies_dict["num_all"] = num_all
        accuracies_dict[key_string + "num_correctly_decoded"] = num_correctly_decoded
        accuracies_dict[key_string + "num_decoded"] = num_decoded
        accuracies_dict[key_string + "decode_rate"] = np.round(num_correctly_decoded / num_all, 4)
        accuracies_dict[key_string + "misread_rate"] = np.round((num_decoded - num_correctly_decoded) / num_decoded, 4)

    
    # "a" means append to file (write just at the end)
    with open(saved_accuracies_path, "a") as file:
        json.dump(accuracies_dict, file)
        file.write("\n")



if __name__ == "__main__":
    app()


# run in terminal:
# $ for ckpt in $(cat checkpoints.txt); do python dm_codes/scripts/measure_accuracy.py $ckpt "checkpoints_accuracies.jsonl"; done
