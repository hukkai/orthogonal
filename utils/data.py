from __future__ import annotations

import os

import torch
from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def build_train_transform(args) -> transforms.Compose:
    return create_transform(
        input_size=args.img_size,
        is_training=True,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        interpolation=args.interpolation,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )


def build_val_transform(args) -> transforms.Compose:
    resize_size = int(args.img_size / args.crop_pct)
    return transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )


def build_imagenet_datasets(args) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    train_dir = os.path.join(args.data, "train")
    val_dir = os.path.join(args.data, "val")
    train_transform = build_train_transform(args)
    val_transform = build_val_transform(args)
    train_set = datasets.ImageFolder(train_dir, transform=train_transform)
    val_set = datasets.ImageFolder(val_dir, transform=val_transform)
    return train_set, val_set