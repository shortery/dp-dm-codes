import typer
import torch
import lightning.pytorch as pl
import time
import timeit
import onnx
import onnxruntime
import numpy as np
import pandas as pd
import tqdm

import my_training

app = typer.Typer(pretty_exceptions_show_locals=False, pretty_exceptions_enable=False)

@app.command()
def main(
    batch_size: int,
    num_batches: int,
    checkpoint_path: str,
    is_onnx: bool = False,
    onnx_model_path: str = "exported_onnx_model.onnx",
    run_on_cpu: bool = False,
    seed: int = 0
) -> None:
    
    torch.manual_seed(seed)
    random_tensors = torch.rand(size=(batch_size * num_batches, 3, 128, 128))

    dataloader_random_tensors = torch.utils.data.DataLoader(
        dataset=[{"image": x} for x in random_tensors],
        batch_size=batch_size
    )

    trainer = pl.Trainer()
    if run_on_cpu:
        loaded_model = my_training.LitAutoEncoder.load_from_checkpoint(checkpoint_path, map_location=torch.device("cpu"))
    else:
        loaded_model = my_training.LitAutoEncoder.load_from_checkpoint(checkpoint_path)
    loaded_model.eval()


    if is_onnx:
        torch.onnx.export(
            loaded_model,
            torch.rand(size=(batch_size, 3, 128, 128)),
            onnx_model_path,
            input_names=["image"],
            output_names = ['output'],
            dynamic_axes={'image' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}}
        )

        onnx_model = onnx.load(onnx_model_path)
        onnx.checker.check_model(onnx_model)
        if run_on_cpu:
            onnx_inference_session = onnxruntime.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
        else:
            onnx_inference_session = onnxruntime.InferenceSession(onnx_model_path, providers=["CUDAExecutionProvider"])

        random_arrays = pd.DataFrame(dataloader_random_tensors).map(np.asarray).to_dict('records')

        start_wall_time = timeit.default_timer()
        start_cpu_time = time.process_time()

        for random_array in tqdm.tqdm(random_arrays):
            onnx_inference_session.run(None, random_array)

        end_wall_time = timeit.default_timer()
        end_cpu_time = time.process_time()


    else:
        start_wall_time = timeit.default_timer()
        start_cpu_time = time.process_time()

        trainer.predict(loaded_model, dataloader_random_tensors)

        end_wall_time = timeit.default_timer()
        end_cpu_time = time.process_time()

    print()
    print(checkpoint_path)
    print("wall time:", round(end_wall_time - start_wall_time, 4), "seconds")
    print("cpu time:", round(end_cpu_time - start_cpu_time, 4), "seconds")
    print()


if __name__ == "__main__":
    app()


# run in terminal:
# $ for ckpt in $(cat checkpoints.txt); do python src/measure_runtime.py 32 100 $ckpt; done
