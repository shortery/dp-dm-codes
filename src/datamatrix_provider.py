from enum import Enum, auto
import os
import random
import time
from typing import Optional

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from ppf.datamatrix.datamatrix import DataMatrix
from pylibdmtx.pylibdmtx import encode

import albumentations


# from src.utils.datamatrix_augmentation import get_datamatrix_augs_preset
# TODO this function has not been provided to me yet
def get_datamatrix_augs_preset(preset_file):
    preserving = albumentations.Resize(80, 80)
    destructive = albumentations.MotionBlur(always_apply=True)
    return preserving, destructive


def visualize_dm_triplet(image, text, augmented_gt_dm_img, distorted_dm_img):
    f, ax = plt.subplots(1, 3)
    ax[0].imshow(image, cmap='gray', vmin=0, vmax=255)
    ax[1].imshow(augmented_gt_dm_img, cmap='gray', vmin=0, vmax=255)
    ax[2].imshow(distorted_dm_img, cmap='gray', vmin=0, vmax=255)
    ax[0].title.set_text("Original DM")
    ax[1].title.set_text("Augmented GT")
    ax[2].title.set_text("Distortion")
    f.suptitle(f"DM code: {text}")
    plt.show()


class DataMatrixProviderType(Enum):
    """Enumeration of types of DataMatrix generators."""
    RandomDMGenerator = auto()
    FileDMGenerator = auto()


class DataMatrixProvider:
    """Iterable that generates DataMatrix code using pylibdmtx. The encoded text is either:
        - random string bounded by length (type RandomDMGenerator)
        - random string from vocabulary file (.data, .txt) with each input on own line. (type FileDMGenerator)
    The output is batch of tuples: QR code grayscale images and ground-truth text.

    Args:
        provider_type (str): Random DM generator (default) or text file generator.
        data_file (str): Required only for DataMatrixProviderType.FileDMGenerator. Defaults to None.
        letters (str): String of allowed characters for DM code generation. Defaults to all ASCII characters.
        min_len (int): Minimum length of text to be encoded in DM code. Defaults to 1.
        max_len (int): Maximum length of text to be encoded in DM code. Defaults to 32.
        batch_size (int): Number of DM codes to be returned. Defaults to 1.
        visualize (bool): Whether to display the DM code. Defaults to False.
        augmentation_preset (str): Path to python file in cfg/augmentation_preset folder. Cfg file defines two types of
            augmentations/transforms: destructive and non-destructive. Defaults to 'default_augmentation.py'.
        crop_pixels (int): Pixels to be cropped from each side of the generated code. Typically, pylibdmtx generates
            DM image with a 10-pixel white border that we want to crop.

    Return:
        list(Tuple(np.ndarray, str)): List of N=batch_size tuples:
            (
                WxH np.ndarray of the DM code image in grayscale,
                string of the ground-truth text
            )
    """

    def __init__(self,
                 library: str = "pylibdmtx",
                 provider_type: DataMatrixProviderType = DataMatrixProviderType.RandomDMGenerator,
                 data_file: Optional[str] = None,
                 letters: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
                 min_len: int = 1,
                 max_len: int = 32,
                 batch_size: int = 1,
                 visualize: bool = False,
                 augmentation_preset: str = "default_augmentation.py",
                 crop_pixels: int = 10,
                 pylibdmtx_params: dict = {
                     "size": 8,
                     "mode": "square",
                     "fat_constant": 0
                 }
                 ):

        if provider_type == DataMatrixProviderType.FileDMGenerator:
            assert data_file is not None, f"data_file must be specified for {DataMatrixProviderType.FileDMGenerator}"
            assert os.path.exists(data_file), f"File {data_file} does not exist."
            raise NotImplementedError("Not implemented yet")
        assert max_len > 0, f"max_len = {max_len} must be greater than 0"

        self.provider_type = provider_type
        self.batch_size = batch_size
        self.visualize = visualize
        self.letters = letters
        self.min_len, self.max_len = min_len, max_len
        self.gt_preserving_augs, self.destructive_augs = get_datamatrix_augs_preset(preset_file=augmentation_preset)
        self.crop_px = crop_pixels
        self.generate_dm, self.block_size, self.mode, self.mode_options, self.fat_constant = self.init_dm_provider(
            library, pylibdmtx_params)

    def __iter__(self):
        return self

    def __next__(self):
        return [self.generate_dm() for _ in range(self.batch_size)]

    def generate_dm_pylibdmtx(self):
        length = random.randint(self.min_len, self.max_len)
        encoded_text = "".join(random.choice(self.letters) for _ in range(length)).encode('utf8')
        data_matrix = encode(encoded_text)
        w, h = data_matrix.width, data_matrix.height

        clean_dm_img = np.asarray(Image.frombytes("RGB", (w, h), data_matrix.pixels).convert("L"))
        clean_dm_img = clean_dm_img[self.crop_px:-self.crop_px, self.crop_px:-self.crop_px]  # crop padding around DM
        augmented_gt_dm_img = self.gt_preserving_augs(image=clean_dm_img)["image"]
        distorted_dm_img = self.destructive_augs(image=augmented_gt_dm_img)["image"]
        if self.visualize:
            visualize_dm_triplet(clean_dm_img, encoded_text, augmented_gt_dm_img, distorted_dm_img)

        return augmented_gt_dm_img, distorted_dm_img, encoded_text

    def generate_dm_ppf_datamatrix(self):
        length = random.randint(self.min_len, self.max_len)
        text = "".join(random.choice(self.letters) for _ in range(length))
        mode = random.choice(self.mode_options) if self.mode == "random" else self.mode
        fat_constant = random.randint(self.fat_constant[0], self.fat_constant[1])
        size = random.randint(self.block_size[0], self.block_size[1])

        data_matrix_arr = DataMatrix(text).matrix
        print(data_matrix_arr)
        height = len(data_matrix_arr)
        width = len(data_matrix_arr[0])
        image = Image.new('L', (size * width, size * height), 255)
        draw = ImageDraw.Draw(image)
        mat_arr = np.asarray(data_matrix_arr)

        offset_x = offset_y = 0
        for i in range(width):
            for j in range(height):
                if mat_arr[i][j] == 1:
                    if mode == 'circle':
                        draw.ellipse((
                            offset_x + j * size + fat_constant,
                            offset_y + i * size + fat_constant,
                            offset_x + (j + 1) * size - fat_constant,
                            offset_y + (i + 1) * size - fat_constant), fill='black', outline='black')
                    elif mode == 'square':
                        draw.rectangle((
                            j * size + fat_constant,
                            i * size + fat_constant,
                            (j + 1) * size - fat_constant,
                            (i + 1) * size - fat_constant), fill='black', outline='black')
                    elif mode == 'triangle':
                        draw.polygon(
                            [(offset_x + (j + 0.5) * size, offset_y + (i * size) + fat_constant),
                             (offset_x + (j + 1) * size - fat_constant, offset_y + (i + 1) * size - fat_constant),
                             (offset_x + j * size + fat_constant, offset_y + (i + 1) * size - fat_constant)],
                            fill='black', outline='black')

        augmented_gt_dm_img = self.gt_preserving_augs(image=np.array(image))["image"]
        distorted_dm_img = self.destructive_augs(image=augmented_gt_dm_img)["image"]

        if self.visualize:
            visualize_dm_triplet(image, text, augmented_gt_dm_img, distorted_dm_img)

        bitmap = np.zeros((32, 32), dtype=np.int)
        bitmap[:mat_arr.shape[0], :mat_arr.shape[1]] = mat_arr

        return augmented_gt_dm_img, distorted_dm_img, text.encode('utf8'), bitmap

    def init_dm_provider(self, library, pylibdmtx_params):
        assert library in ["pylibdmtx", "ppf-datamatrix"], 'Supported libraries are "pylibdmtx", "ppf-datamatrix".'
        generate_function = self.generate_dm_pylibdmtx if library == "pylibdmtx" else self.generate_dm_ppf_datamatrix
        block_size = pylibdmtx_params.get("size", 10)
        if isinstance(block_size, int):
            block_size = (block_size, block_size)
        fat_constant = pylibdmtx_params.get("fat_constant", 0)
        if isinstance(fat_constant, int):
            fat_constant = (fat_constant, fat_constant)
        assert isinstance(fat_constant, list) or isinstance(fat_constant, tuple)
        assert all(block_size) > 0, f"Block size {block_size} must be > 0."
        assert block_size[0] <= block_size[1]
        assert fat_constant[0] <= fat_constant[1]

        mode_options = ["random", "square", "circle", "triangle"]
        mode = pylibdmtx_params.get("mode", "square")
        assert mode in mode_options, f"Mode {mode} not supported, use one from: {mode_options}."
        mode_options.remove("random")
        return generate_function, block_size, mode, mode_options, fat_constant



# PERFORMANCE BENCHMARK
if __name__ == "__main__":
    dm_provider = DataMatrixProvider(
        library="ppf-datamatrix",
        visualize=True,
        augmentation_preset="augmentation_v2.py",
        pylibdmtx_params={
            "size": 10,
            "mode": "random",
            "fat_constant": [-1, 1]
        }
    )
    batch_size = dm_provider.batch_size
    t = time.time()
    for idx, a in enumerate(dm_provider):
        if idx > int(1000 / batch_size):
            break
    print(f"Mean time to generate 1 DM code: {round(1000 * (time.time() - t) / (idx * batch_size), 6)} ms")
