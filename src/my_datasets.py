import numpy as np
import pandas as pd
import random
import torch
from PIL import Image
import pathlib
import datasets
from skimage import transform

import my_datamatrix_provider

def _preprocess(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        new_shape = (1,) + image.shape
        image = image.reshape(new_shape)
    if (image.ndim == 3) & (image.shape[2] == 3):
        w, h, num_channels = image.shape
        image = np.transpose(image, (2, 0, 1))
        assert image.shape == (num_channels, w, h) 
    return image.astype(np.float32) / 255

def create_dataset(num_samples: int, seed: int) -> pd.DataFrame:
    random.seed(seed)
    dataset = []
    dm_provider = my_datamatrix_provider.DataMatrixProvider()
    for _ in range(num_samples):
        dm_target, dm_corrupt, dm_text = next(dm_provider)[0]
        dataset.append((dm_target, dm_corrupt, dm_text))
    return pd.DataFrame(dataset, columns=["target", "corrupted", "text"])


class MyMapDataset(torch.utils.data.Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, i: int) -> dict:
        return {"target": _preprocess(self.dataset.iloc[i]["target"]),
                "corrupted": _preprocess(self.dataset.iloc[i]["corrupted"]),
                "text": self.dataset.iloc[i]["text"].decode("utf8")}
    

class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, dm_provider: my_datamatrix_provider.DataMatrixProvider) -> None:
        super().__init__()
        self.dm_provider = dm_provider

    def __iter__(self):
        while True:
            batch = next(self.dm_provider)
            for dm_target, dm_corrupt, dm_text in batch:
                yield {"target": _preprocess(dm_target),
                       "corrupted": _preprocess(dm_corrupt),
                       "text": dm_text.decode("utf8")}
                

class MyMapDatasetFromFolder(torch.utils.data.Dataset):
    def __init__(self, folder, limit: int = -1) -> None:
        super().__init__()
        self.folder = folder
        self.limit = limit

    def __len__(self) -> int:
        num_files = len(list(pathlib.Path(self.folder).glob("text/*.txt")))
        if self.limit > 0:
            return min(num_files, self.limit)
        return num_files

    def __getitem__(self, i: int) -> dict:
        with open(f"{self.folder}/text/{i}.txt", "r") as file:
            text = file.readline()
        target_image = Image.open(f"{self.folder}/target/{i}.png")
        corrupted_image = Image.open(f"{self.folder}/corrupted/{i}.png")
        return {"target": _preprocess(np.array(target_image)),
                "corrupted": _preprocess(np.array(corrupted_image)),
                "text": text}
                

class MyMapDatasetFromHuggingFace(torch.utils.data.Dataset):
    def __init__(self, hf_dataset: datasets.Dataset) -> None:
        super().__init__()
        self.random_generator = random.Random(0)
        df = pd.DataFrame(hf_dataset)
        df["image"] = df["image"].apply(self.preprocess_image)
        self.df = df

    def preprocess_image(self, image: Image.Image):
        image = image.resize((128, 128), resample=Image.Resampling.NEAREST)
        angle = self.random_generator.randint(0, 3) * 90
        image = image.rotate(angle)
        return _preprocess(np.asarray(image))

    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, i: int) -> dict:
        return {"image": self.df.iloc[i]["image"],
                "text": self.df.iloc[i]["text"]}
    

def crop_dm_code(example: dict, square_side: int = 200, square_padding: int = 25) -> dict:
    vertices = np.asarray((example["tl"], example["tr"], example["br"], example["bl"]))
    unit_square = np.asarray([
        [square_padding, square_padding],
        [square_side + square_padding, square_padding],
        [square_side + square_padding, square_side + square_padding],
        [square_padding, square_side + square_padding]
    ])
    transf = transform.ProjectiveTransform()
    if not transf.estimate(unit_square, vertices): raise Exception("estimate failed")
    cropped_np_image = transform.warp(
        np.array(example["image"]),
        transf,
        output_shape=(square_side + square_padding * 2, square_side + square_padding * 2)
    )
    cropped_image = Image.fromarray((cropped_np_image * 255).astype(np.uint8))
    return {"image": cropped_image}
