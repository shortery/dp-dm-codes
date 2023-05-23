import numpy as np
import torch

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
    image_array = image_array.reshape(batch_size, w, h, num_channels)    

    return image_array
