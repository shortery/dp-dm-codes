import random
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import pylibdmtx.pylibdmtx
import string

import dm_codes.augmentations

def visualize_images(text, target_img, corrupted_img):
    f, ax = plt.subplots(1, 2)
    ax[0].imshow(target_img, cmap='gray', vmin=0, vmax=255)
    ax[1].imshow(corrupted_img, cmap='gray', vmin=0, vmax=255)
    ax[0].title.set_text("Target DM code")
    ax[1].title.set_text("Corrupted DM code")
    for a in ax:
        a.axis('off')
    f.suptitle(f"DM code: {text}")
    plt.tight_layout()
    plt.show()


class DataMatrixProvider:
    """Iterable that generates DataMatrix code using pylibdmtx."""
    def __init__(self,
        letters: str = string.ascii_letters + string.digits,
        min_len: int = 1,
        max_len: int = 32,
        batch_size: int = 1,
        visualize: bool = False
        ):

        self.letters = letters
        self.min_len = min_len
        self.max_len = max_len
        self.batch_size = batch_size
        self.visualize = visualize

    def __iter__(self):
        return self

    def __next__(self):
        return [self.generate_dm_pylibdmtx() for _ in range(self.batch_size)]
    
    def generate_dm_pylibdmtx(self):
        length = random.randint(self.min_len, self.max_len)
        encoded_text = "".join(random.choice(self.letters) for _ in range(length)).encode('utf8')
        dm_code = pylibdmtx.pylibdmtx.encode(encoded_text)
        w, h = dm_code.width, dm_code.height
    
        preserving_augs, destructive_augs = dm_codes.augmentations.get_datamatrix_augs_preset()
        clean_image = np.asarray(Image.frombytes("RGB", (w, h), dm_code.pixels).convert("L"))
        target_image =preserving_augs(image=clean_image)["image"]
        corrupted_image = destructive_augs(image=target_image)["image"]
        if self.visualize:
            visualize_images(encoded_text, target_image, corrupted_image)

        return target_image, corrupted_image, encoded_text
        

