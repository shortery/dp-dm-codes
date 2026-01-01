from PIL import Image
import os
import typer

import dm_codes.datasets 

app = typer.Typer(pretty_exceptions_show_locals=False, pretty_exceptions_enable=False)

@app.command()
def main(
    image_dir: str = "datasets/synthetic_valid_dataset",
    dataset_size: int = 100,
    dataset_seed: int = 0,
) -> None:

    pd_dataset = dm_codes.datasets.create_dataset(dataset_size, dataset_seed)

    os.makedirs(image_dir, exist_ok=False)
    for col in pd_dataset.columns:
        os.makedirs(f"{image_dir}/{col}", exist_ok=False)

    for i in range(dataset_size):
        for col in pd_dataset.columns:
            file_name = f"{image_dir}/{col}/{i}"
            if col == "text":
                with open(file_name + ".txt", "w") as file:
                    file.write(pd_dataset[col][i].decode("utf8"))
            else:
                Image.fromarray(pd_dataset[col][i]).save(file_name + ".png")
            

if __name__ == "__main__":
    app()