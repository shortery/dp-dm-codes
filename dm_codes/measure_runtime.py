import statistics
import time
import timeit
import json

import typer
import torch
import lightning.pytorch as pl
import logging
import onnx
import onnxruntime
import numpy as np
import pandas as pd
import tqdm

import dm_codes.training

app = typer.Typer(pretty_exceptions_show_locals=False, pretty_exceptions_enable=False)

@app.command()
def main(
    num_examples: int,
    checkpoint_path: str,
    saved_runtimes_path: str,
    is_onnx: bool = False,
    run_on_cpu: bool = False,
    onnx_model_path: str = "exported_onnx_model.onnx",
    seed: int = 0,
    batch_size: int = 1
) -> None:
    
    torch.set_float32_matmul_precision('high')
    logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)
    torch.manual_seed(seed)
    device = 'cpu' if run_on_cpu else 'cuda'
    random_tensors = torch.rand(size=(num_examples, 3, 128, 128)).to(device)

    dataloader_random_tensors = torch.utils.data.DataLoader(
        dataset=[{"image": x} for x in random_tensors],
        batch_size=batch_size
    )

    loaded_model = dm_codes.training.LitAutoEncoder.load_from_checkpoint(checkpoint_path, map_location=device)
    loaded_model.eval()

    list_wall_times = []
    list_process_times = []

    if is_onnx:
        torch.onnx.export(
            loaded_model,
            torch.rand(size=(batch_size, 3, 128, 128)).to(device),
            onnx_model_path,
            input_names=["image"],
            output_names = ['output'],
            dynamic_axes={'image' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}}
        )

        onnx_model = onnx.load(onnx_model_path)
        onnx.checker.check_model(onnx_model)

        providers=["CPUExecutionProvider"] if run_on_cpu else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        onnx_inference_session = onnxruntime.InferenceSession(onnx_model_path, providers=providers)

        random_arrays = pd.DataFrame(dataloader_random_tensors).map(lambda x: onnxruntime.OrtValue.ortvalue_from_numpy(x.cpu().numpy(), device, 0)).to_dict('records')

        onnx_inference_session.run(None, random_arrays[0]) # first slow prediction

        for random_array in tqdm.tqdm(random_arrays):
            start_wall_time = timeit.default_timer()
            start_process_time = time.process_time()

            onnx_inference_session.run(None, random_array)

            end_wall_time = timeit.default_timer()
            end_process_time = time.process_time()

            list_wall_times.append(round(end_wall_time - start_wall_time, 6))
            list_process_times.append(round(end_process_time - start_process_time, 6))


    else:

        loaded_model(next(iter(dataloader_random_tensors))["image"]) # first slow prediction

        for batch in tqdm.tqdm(dataloader_random_tensors):
            start_wall_time = timeit.default_timer()
            start_process_time = time.process_time()

            loaded_model(batch["image"]) # predict

            end_wall_time = timeit.default_timer()
            end_process_time = time.process_time()

            list_wall_times.append(round(end_wall_time - start_wall_time, 6))
            list_process_times.append(round(end_process_time - start_process_time, 6))


    runtimes_dict = {
        "checkpoint_path": checkpoint_path,
        "num_examples": num_examples,
        "is_onnx": is_onnx,
        "run_on_cpu": run_on_cpu,
        "wall_time": list_wall_times,
        "process_time": list_process_times,
        "median_wall_time": round(statistics.median(list_wall_times), 6),
        "median_process_time": round(statistics.median(list_process_times), 6),
        "loaded_model_hparams": loaded_model.hparams
    }
    # "a" means append to file (write just at the end)
    with open(saved_runtimes_path, "a") as file:
        json.dump(runtimes_dict, file)
        file.write("\n")



if __name__ == "__main__":
    app()


# run in terminal:
# $ for ckpt in $(cat checkpoints.txt); do python dm_codes/measure_runtime.py 100 $ckpt "checkpoints_runtimes.jsonl"; sleep 2; done
