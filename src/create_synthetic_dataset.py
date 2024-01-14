from PIL import Image
import os

import my_datasets 

valid_size = 1000
valid_seed = 0

pd_dataset = my_datasets.create_dataset(valid_size, valid_seed)

image_dir = "./synthetic_valid_dataset"
os.makedirs(image_dir, exist_ok=False)
for col in pd_dataset.columns:
    os.makedirs(f"{image_dir}/{col}", exist_ok=False)

for i in range(valid_size):
    for col in pd_dataset.columns:
        file_name = f"{image_dir}/{col}/{i}"
        if col == "text":
            with open(file_name + ".txt", "w") as file:
                file.write(pd_dataset[col][i].decode("utf8"))
        else:
            Image.fromarray(pd_dataset[col][i]).save(file_name + ".png")
            