# modules/transforms.py

from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageOps


class LetterBox:
    """
    Resize an image while preserving aspect ratio, then pad it to a square.
    """

    def __init__(self, size: int, fill: tuple[int, int, int] = (0, 0, 0)):
        self.size = int(size)
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
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
    """

    def __init__(self, fill: tuple[int, int, int] = (0, 0, 0)):
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
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
    """

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return mean, std


def get_normalize_transform() -> T.Normalize:
    """
    Return ImageNet normalization transform.
    """

    mean, std = get_imagenet_mean_std()
    return T.Normalize(mean=mean, std=std)


def get_hybrid_transform(
    mode: str,
    image_size: int = 224,
    train: bool = False,
) -> T.Compose:
    """
    Return transform for hybrid feature extraction.

    Supported modes:
    - stretch
    - center_crop
    - letterbox
    - augmented
    """

    mode = mode.lower().strip()
    normalize = get_normalize_transform()

    if mode == "stretch":
        return T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                normalize,
            ]
        )

    if mode == "center_crop":
        return T.Compose(
            [
                T.Resize(int(image_size * 1.14)),
                T.CenterCrop(image_size),
                T.ToTensor(),
                normalize,
            ]
        )

    if mode == "letterbox":
        return T.Compose(
            [
                LetterBox(image_size),
                T.ToTensor(),
                normalize,
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
                    T.ToTensor(),
                    normalize,
                ]
            )

        return T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                normalize,
            ]
        )

    raise ValueError(
        f"Unsupported transform mode: {mode}. "
        "Supported modes: stretch, center_crop, letterbox, augmented."
    )


def get_dl_transform(
    image_size: int = 224,
    train: bool = False,
) -> T.Compose:
    """
    Return transform for end-to-end deep learning.
    """

    normalize = get_normalize_transform()

    if train:
        return T.Compose(
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

    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            normalize,
        ]
    )


def tensor_to_display_image(
    tensor: torch.Tensor,
    denormalize: bool = True,
) -> np.ndarray:
    """
    Convert a normalized image tensor to a displayable NumPy image.
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