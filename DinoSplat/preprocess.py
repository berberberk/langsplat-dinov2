import os
import random
import argparse
from typing import Iterable

from pathlib import Path

import numpy as np
import cv2

import torch
from torch import nn
from torchvision import transforms


device = "cuda" if torch.cuda.is_available() else "cpu"

image_transforms = transforms.Compose(
    [
        transforms.Resize((518, 518)),
        lambda x: x / 255.0,
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


@torch.no_grad()
def create(
    model: nn.Module,
    image_list: Iterable[torch.Tensor],
    data_list: Iterable[str],
    save_folder: str,
):
    model.to(device)
    save_dir = Path(save_folder)
    for image, file_path in zip(image_list, data_list):
        image = image_transforms(image.to(device)[None])
        result = (
            model.forward_features(image)["x_norm_patchtokens"]
            .view(37, 37, -1)
            .permute(2, 0, 1)
            .cpu()
            .numpy()
        )

        save_path = save_dir / f"{Path(file_path).stem}.npz"

        np.savez_compressed(save_path, result)


def seed_everything(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def main():
    seed_num = 42
    seed_everything(seed_num)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=-1)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)

    dataset_path = args.dataset_path
    img_folder = os.path.join(dataset_path, "images")
    data_list = os.listdir(img_folder)
    data_list.sort()

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")

    img_list = []
    for data_path in data_list:
        image_path = os.path.join(img_folder, data_path)

        image = cv2.imread(image_path)
        image = torch.from_numpy(image).flip(dims=[-1])

        img_list.append(image)

    images = [img_list[i].permute(2, 0, 1)[None, ...] for i in range(len(img_list))]
    imgs = torch.cat(images)

    save_folder = os.path.join(dataset_path, "dino_features")
    os.makedirs(save_folder, exist_ok=True)

    create(model, imgs, data_list, save_folder)


if __name__ == "__main__":
    main()
