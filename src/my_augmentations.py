import warnings
import albumentations
import cv2

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



def get_datamatrix_augs_preset():
    preserving = albumentations.Resize(80, 80, interpolation=cv2.INTER_LANCZOS4)
    destructive = albumentations.Compose([
        albumentations.CoarseDropout(fill_value=0, max_height=2, max_width=2, max_holes=40),
        albumentations.CoarseDropout(fill_value=255, max_height=2, max_width=2, max_holes=40),
        albumentations.OneOf([
            albumentations.MotionBlur(blur_limit=3),
            albumentations.MedianBlur(blur_limit=3),
            albumentations.Defocus(radius=1),
        ], p=0.75),
        ToRGB(always_apply=True),
        albumentations.Spatter(),
        albumentations.Downscale(interpolation=cv2.INTER_LANCZOS4),
        albumentations.RandomShadow(),
        albumentations.RandomSunFlare(src_radius=60),
        ToGrey(always_apply=True),
    ])
    return preserving, destructive
