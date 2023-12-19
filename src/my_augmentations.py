import warnings
import albumentations
import cv2
import numpy as np
import random

# Taken from: https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/transforms.py
# because it is not released on pypi yet
class ToRGB(albumentations.ImageOnlyTransform):
    """Convert the input grayscale image to RGB.

    Args:
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, always_apply=True, p=1.0):
        super(ToRGB, self).__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        if albumentations.is_rgb_image(img):
            warnings.warn("The image is already an RGB.")
            return img
        if not albumentations.is_grayscale_image(img):
            raise TypeError("ToRGB transformation expects 2-dim images or 3-dim with the last dimension equal to 1.")

        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    def get_transform_init_args_names(self):
        return ()
    

class ToGrey(albumentations.ImageOnlyTransform):
    """Convert RGB image to grayscale."""

    def __init__(self, always_apply=True, p=1.0):
        super(ToGrey, self).__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        if albumentations.is_grayscale_image(img):
            warnings.warn("The image is already grayscale.")
            return img
        if not albumentations.is_rgb_image(img):
            raise TypeError("ToGray transformation expects 3-channel images.")

        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def get_transform_init_args_names(self):
        return ()
    


class ChangeBackgroundColor(albumentations.ImageOnlyTransform):
    """Change background white pixels to random light color"""

    def __init__(self, always_apply=True, p=1.0):
        super(ChangeBackgroundColor, self).__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        if not albumentations.is_rgb_image(img):
            raise TypeError("ChangeBackgroundColor transformation expects 3-channel images.")
        
        synth_img = np.array(img)
        red, green, blue = synth_img.T
        white_areas = (red == 255) & (blue == 255) & (green == 255)
        if random.random() <= 0.5:
            random_color = random.sample(range(190, 250), 3) # light color
        else: 
            random_color = [random.randrange(170, 252)] * 3 # shade of grey
        synth_img[white_areas.T] = random_color
        return synth_img

    def get_transform_init_args_names(self):
        return ()



def get_datamatrix_augs_preset():
    preserving = albumentations.Compose([
        albumentations.Resize(128, 128, interpolation=cv2.INTER_NEAREST),
        albumentations.RandomRotate90(),
        albumentations.Rotate(limit=[-3, 3], border_mode=cv2.BORDER_CONSTANT, value = [255, 255, 255])
    ], p=1)
    destructive = albumentations.Compose([
        ToRGB(always_apply=True),
        ChangeBackgroundColor(always_apply=True),
        albumentations.ISONoise(intensity=(0.5, 0.9), p=0.7),
        albumentations.OneOf([
            albumentations.MotionBlur(blur_limit=(9, 15), p=1, allow_shifted=False),
            albumentations.Defocus(radius=(3, 7), p=1),
            albumentations.Downscale(interpolation=cv2.INTER_LANCZOS4, p=1),
        ], p=1),
        albumentations.Spatter(intensity=0.6, p=0.7),
        albumentations.RandomSunFlare(src_radius=80, num_flare_circles_lower=3, p=0.35)
    ])
    return preserving, destructive

