import numpy as np
import torch
import pylibdmtx.pylibdmtx

def image_to_tensor():
    # torchvision.transforms.functional.pil_to_tensor
    pass

def tensor_to_numpy_for_image(tensor: torch.Tensor) -> np.ndarray:
    batch_size, num_channels, w, h = tensor.shape
    if num_channels != 1:
        raise ValueError('Number of channels must be 1.')
    
    image_array = tensor.clone().cpu().numpy()
    image_array = image_array * 255
    image_array = np.clip(image_array, 0, 255)
    image_array = image_array.astype(np.uint8)

    # wands.Image expects num_channels as the last dimension
    # that's standard order for numpy array 
    image_array = image_array.reshape(batch_size, w, h, num_channels)    

    return image_array


def compute_metrics(target: torch.Tensor, pred: torch.Tensor, text: list[str], prefix: str = None):
    mse_loss = torch.nn.functional.mse_loss(target, pred)

    target_array = tensor_to_numpy_for_image(target)
    pred_array = tensor_to_numpy_for_image(pred)

    close_pixels = np.mean(np.isclose(target_array, pred_array, atol=5))

    decodable = 0
    correctly_decoded = 0
    batch_size, num_channels, w, h = pred.shape
    assert num_channels == 1

    for i in range(batch_size):
        pred_array_i = pred_array[i].reshape(w, h)
        decoded_pred_i = pylibdmtx.pylibdmtx.decode(pred_array_i)
        if len(decoded_pred_i) != 0:
            decodable += 1
            if text[i] == decoded_pred_i[0].data.decode("utf8"):
                correctly_decoded += 1

    if prefix is None:
        prefix = ""

    metrics = {
        prefix + "mse_loss": mse_loss,
        prefix + "close_pixels": close_pixels,
        prefix + "decodable": decodable / batch_size,
        prefix + "correctly_decoded": correctly_decoded / batch_size
    }

    return metrics
