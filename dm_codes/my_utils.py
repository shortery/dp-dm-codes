import numpy as np
import torch
import PIL.Image, PIL.ImageOps
import pylibdmtx.pylibdmtx
import lovely_tensors
import lovely_numpy
import warnings
import concurrent.futures

def image_to_tensor():
    # torchvision.transforms.functional.pil_to_tensor
    pass

def tensor_to_numpy_for_image(tensor: torch.Tensor) -> np.ndarray:
    batch_size, num_channels, w, h = tensor.shape
    
    image_array = tensor.detach().clone().cpu().numpy()
    image_array = image_array * 255
    image_array = np.clip(image_array, 0, 255)
    image_array = image_array.astype(np.uint8)

    # wands.Image expects num_channels as the last dimension
    # that's standard order for numpy array 
    image_array = np.transpose(image_array, (0, 2, 3, 1))
    assert image_array.shape == (batch_size, w, h, num_channels)    

    return image_array


def decode_dm_code(dm_code: np.ndarray):
    "if possible, decode dm code array to text"
    assert len(dm_code.shape) == 2

    dm_code_padded = np.pad(dm_code, 5, mode='constant', constant_values=255) # add 5-pixel width border of white pixels

    # pylibdmtx sometimes causes the whole process to crash without throwing an exception
    # run function in subprocess so it is safe
    try:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future = executor.submit(pylibdmtx.pylibdmtx.decode, dm_code_padded)
            decoded_dm_code = future.result()
    except concurrent.futures.process.BrokenProcessPool as e:
        print(f"Caught BrokenProcessPool exception while decoding image: {repr(e)}")
        print(f"{dm_code_padded.min()=}, {dm_code_padded.max()=}")
        return None
    except Exception as e:
        print(f"Caught exception while decoding image: {repr(e)}")
        return None
    
    if len(decoded_dm_code) == 0:
        return None
    try:
        decoded_text = decoded_dm_code[0].data.decode("utf8")
        return decoded_text
    except UnicodeDecodeError:
        return None
    except Exception as e:
        print(f"Caught exception while decoding text: {repr(e)}")
        return None


def compute_metrics(pred: torch.Tensor, text: list[str], target: torch.Tensor = None, prefix: str = None):
    pred_array = tensor_to_numpy_for_image(pred)
    decodable = 0
    correctly_decoded = 0
    batch_size, num_channels, w, h = pred.shape
    assert num_channels == 1

    for i in range(batch_size):
        pred_array_i = pred_array[i].reshape(w, h)
        decoded_text_i = decode_dm_code(pred_array_i)
        if decoded_text_i is not None:
            decodable += 1
            if text[i] == decoded_text_i:
                correctly_decoded += 1

    if prefix is None:
        prefix = ""

    metrics = {
        prefix + "decodable": decodable / batch_size,
        prefix + "correctly_decoded": correctly_decoded / batch_size
    }

    if target is not None:
        mse_loss: float = torch.nn.functional.mse_loss(target, pred).item()
        metrics[prefix + "mse_loss"] = mse_loss

    return metrics


def lovely(x):
    "summarize important numpy/tensor properties"
    if isinstance(x, np.ndarray):
        return lovely_numpy.lovely(x)
    elif isinstance(x, torch.Tensor):
        return lovely_tensors.lovely(x)
    else:
        warnings.warn(f"lovely: unknown type {type(x)}")
        return str(x)
