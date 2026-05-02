from PIL import Image
import torchvision.transforms as T


class LetterBox:
    """
    Resize image while preserving aspect ratio, then pad to square size.
    """
    def __init__(self, size: int, fill=(0, 0, 0)):
        ...

    def __call__(self, img: Image.Image) -> Image.Image:
        ...


def get_imagenet_mean_std() -> tuple[list[float], list[float]]:
    """
    Return ImageNet mean and standard deviation.
    """
    ...


def get_normalize_transform():
    """
    Return torchvision Normalize transform using ImageNet statistics.
    """
    ...


def get_hybrid_transform(
    mode: str,
    image_size: int = 224,
    train: bool = False
):
    """
    Return transform for hybrid feature extraction.
    Supported modes:
    stretch, center_crop, letterbox, augmented.
    """
    ...


def get_dl_transform(
    image_size: int = 224,
    train: bool = False
):
    """
    Return transform for end-to-end deep learning.
    train=True applies augmentation.
    train=False applies deterministic transform.
    """
    ...


def get_transform_preview(
    image_path: str,
    modes: list[str],
    image_size: int = 224
):
    """
    Apply multiple transforms to one image for visualization.
    Return list of transformed PIL images or tensors converted to images.
    """
    ...