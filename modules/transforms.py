# modules/transforms.py

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageOps


class LetterBox:
    """
    Resize an image while preserving its aspect ratio, then pad it to a square.

    This transformation is useful when geometric distortion should be avoided.
    The output image has size (size, size).
    """

    def __init__(self, size: int, fill: tuple[int, int, int] = (0, 0, 0)):
        """
        Parameters
        ----------
        size : int
            Target square size.
        fill : tuple[int, int, int]
            RGB padding colour.
        """

        self.size = int(size)
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply letterbox resizing.

        Parameters
        ----------
        img : PIL.Image.Image
            Input image.

        Returns
        -------
        PIL.Image.Image
            Letterboxed image with size (size, size).
        """

        img = img.convert("RGB")
        width, height = img.size

        if width <= 0 or height <= 0:
            return Image.new("RGB", (self.size, self.size), self.fill)

        scale = min(self.size / width, self.size / height)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))

        resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (self.size, self.size), self.fill)

        left = (self.size - new_width) // 2
        top = (self.size - new_height) // 2

        canvas.paste(resized, (left, top))

        return canvas


class SquarePad:
    """
    Pad an image to a square shape without resizing.

    This can be useful for visualization or before applying Resize.
    """

    def __init__(self, fill: tuple[int, int, int] = (0, 0, 0)):
        """
        Parameters
        ----------
        fill : tuple[int, int, int]
            RGB padding colour.
        """

        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply square padding.

        Parameters
        ----------
        img : PIL.Image.Image
            Input image.

        Returns
        -------
        PIL.Image.Image
            Padded square image.
        """

        img = img.convert("RGB")
        width, height = img.size

        max_side = max(width, height)

        pad_left = (max_side - width) // 2
        pad_top = (max_side - height) // 2
        pad_right = max_side - width - pad_left
        pad_bottom = max_side - height - pad_top

        return ImageOps.expand(
            img,
            border=(pad_left, pad_top, pad_right, pad_bottom),
            fill=self.fill,
        )


def get_imagenet_mean_std() -> tuple[list[float], list[float]]:
    """
    Return ImageNet mean and standard deviation.

    These values should be used when feeding images into pretrained
    torchvision backbones.

    Returns
    -------
    tuple[list[float], list[float]]
        ImageNet mean and standard deviation.
    """

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return mean, std


def get_normalize_transform() -> T.Normalize:
    """
    Return torchvision normalization transform using ImageNet statistics.

    Returns
    -------
    torchvision.transforms.Normalize
        ImageNet normalization transform.
    """

    mean, std = get_imagenet_mean_std()

    return T.Normalize(mean=mean, std=std)


def get_hybrid_transform(
    mode: str,
    image_size: int = 224,
    train: bool = False,
) -> T.Compose:
    """
    Return image transformation for the hybrid feature-extraction pipeline.

    Supported modes
    ---------------
    stretch:
        Directly resize image to (image_size, image_size).
    center_crop:
        Resize image to a slightly larger scale, then center crop.
    letterbox:
        Preserve aspect ratio and pad image to square.
    augmented:
        Resize image and apply light augmentation when train=True.

    Parameters
    ----------
    mode : str
        Transformation mode.
    image_size : int
        Target image size.
    train : bool
        Whether to apply stochastic training augmentation.

    Returns
    -------
    torchvision.transforms.Compose
        Image transform.
    """

    mode = mode.lower().strip()
    normalize = get_normalize_transform()

    if mode == "stretch":
        transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                normalize,
            ]
        )

    elif mode == "center_crop":
        transform = T.Compose(
            [
                T.Resize(int(image_size * 1.14)),
                T.CenterCrop(image_size),
                T.ToTensor(),
                normalize,
            ]
        )

    elif mode == "letterbox":
        transform = T.Compose(
            [
                LetterBox(image_size),
                T.ToTensor(),
                normalize,
            ]
        )

    elif mode == "augmented":
        if train:
            transform = T.Compose(
                [
                    T.Resize((image_size, image_size)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                    ),
                    T.ToTensor(),
                    normalize,
                ]
            )
        else:
            transform = T.Compose(
                [
                    T.Resize((image_size, image_size)),
                    T.ToTensor(),
                    normalize,
                ]
            )

    else:
        raise ValueError(
            f"Unsupported hybrid transform mode: {mode}. "
            "Supported modes: stretch, center_crop, letterbox, augmented."
        )

    return transform


def get_dl_transform(
    image_size: int = 224,
    train: bool = False,
) -> T.Compose:
    """
    Return image transformation for end-to-end deep learning.

    Training transform includes augmentation.
    Evaluation transform is deterministic.

    Parameters
    ----------
    image_size : int
        Target image size.
    train : bool
        Whether to return training transform.

    Returns
    -------
    torchvision.transforms.Compose
        Image transform.
    """

    normalize = get_normalize_transform()

    if train:
        transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                ),
                T.ToTensor(),
                normalize,
            ]
        )
    else:
        transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                normalize,
            ]
        )

    return transform


def get_preview_transform(
    mode: str,
    image_size: int = 224,
    train: bool = False,
) -> Callable[[Image.Image], Image.Image]:
    """
    Return a PIL-based transform for visual preview.

    This function does not apply tensor conversion or normalization, so the
    output can be directly displayed with matplotlib.

    Parameters
    ----------
    mode : str
        Transformation mode.
    image_size : int
        Target image size.
    train : bool
        Whether to apply stochastic augmentation for preview.

    Returns
    -------
    Callable
        Transform that returns a PIL image.
    """

    mode = mode.lower().strip()

    if mode == "original":
        return lambda img: img.convert("RGB")

    if mode == "stretch":
        return T.Compose(
            [
                T.Resize((image_size, image_size)),
            ]
        )

    if mode == "center_crop":
        return T.Compose(
            [
                T.Resize(int(image_size * 1.14)),
                T.CenterCrop(image_size),
            ]
        )

    if mode == "letterbox":
        return LetterBox(image_size)

    if mode == "square_pad":
        return T.Compose(
            [
                SquarePad(),
                T.Resize((image_size, image_size)),
            ]
        )

    if mode == "augmented":
        if train:
            return T.Compose(
                [
                    T.Resize((image_size, image_size)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                    ),
                ]
            )

        return T.Compose(
            [
                T.Resize((image_size, image_size)),
            ]
        )

    raise ValueError(
        f"Unsupported preview transform mode: {mode}. "
        "Supported modes: original, stretch, center_crop, letterbox, "
        "square_pad, augmented."
    )


def tensor_to_display_image(
    tensor: torch.Tensor,
    denormalize: bool = True,
) -> np.ndarray:
    """
    Convert a tensor image into a displayable NumPy image.

    Parameters
    ----------
    tensor : torch.Tensor
        Image tensor with shape (C, H, W).
    denormalize : bool
        Whether to reverse ImageNet normalization.

    Returns
    -------
    np.ndarray
        Image array with shape (H, W, C), values in [0, 1].
    """

    if not torch.is_tensor(tensor):
        raise TypeError("Input must be a torch.Tensor.")

    img = tensor.detach().cpu().clone()

    if img.ndim != 3:
        raise ValueError("Expected tensor shape (C, H, W).")

    if denormalize:
        mean, std = get_imagenet_mean_std()
        mean = torch.tensor(mean).view(3, 1, 1)
        std = torch.tensor(std).view(3, 1, 1)
        img = img * std + mean

    img = img.clamp(0, 1)
    img = img.permute(1, 2, 0).numpy()

    return img


def pil_to_numpy(img: Image.Image) -> np.ndarray:
    """
    Convert PIL image to NumPy array.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image.

    Returns
    -------
    np.ndarray
        RGB image array.
    """

    return np.asarray(img.convert("RGB"))


def apply_preview_transforms(
    image_path: str,
    modes: list[str],
    image_size: int = 224,
    train: bool = False,
) -> list[tuple[str, Image.Image]]:
    """
    Apply multiple preview transformations to one image.

    Parameters
    ----------
    image_path : str
        Path to image.
    modes : list[str]
        List of preview transform modes.
    image_size : int
        Target image size.
    train : bool
        Whether stochastic augmentation is enabled for preview.

    Returns
    -------
    list[tuple[str, PIL.Image.Image]]
        List of (mode, transformed_image).
    """

    img = Image.open(image_path).convert("RGB")
    outputs = []

    for mode in modes:
        transform = get_preview_transform(
            mode=mode,
            image_size=image_size,
            train=train,
        )
        transformed = transform(img)
        outputs.append((mode, transformed.convert("RGB")))

    return outputs


def get_transform_grid(
    image_paths: list[str],
    modes: list[str],
    image_size: int = 224,
    train: bool = False,
) -> list[dict]:
    """
    Apply multiple transformations to multiple images.

    This function is mainly used before visualization.

    Parameters
    ----------
    image_paths : list[str]
        List of image paths.
    modes : list[str]
        Transformation modes.
    image_size : int
        Target image size.
    train : bool
        Whether stochastic augmentation is enabled.

    Returns
    -------
    list[dict]
        Each dictionary contains path, mode, and image.
    """

    records = []

    for path in image_paths:
        transformed_images = apply_preview_transforms(
            image_path=path,
            modes=modes,
            image_size=image_size,
            train=train,
        )

        for mode, img in transformed_images:
            records.append(
                {
                    "path": path,
                    "mode": mode,
                    "image": img,
                }
            )

    return records


def get_available_transform_modes() -> list[str]:
    """
    Return supported transformation modes for the hybrid pipeline.

    Returns
    -------
    list[str]
        Supported modes.
    """

    return [
        "stretch",
        "center_crop",
        "letterbox",
        "augmented",
    ]


def validate_transform_mode(mode: str) -> None:
    """
    Validate whether a transform mode is supported.

    Parameters
    ----------
    mode : str
        Transform mode.

    Raises
    ------
    ValueError
        If mode is not supported.
    """

    available = get_available_transform_modes()

    if mode not in available:
        raise ValueError(
            f"Invalid transform mode: {mode}. "
            f"Available modes: {available}"
        )