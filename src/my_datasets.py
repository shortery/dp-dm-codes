import numpy as np
import pandas as pd
import random
import torch

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
                
