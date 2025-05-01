import os
import random
import argparse
import sys
from dataclasses import dataclass, field
from typing import Tuple, Type
from copy import deepcopy
from pathlib import Path
from omegaconf import OmegaConf

import numpy as np
import torch
from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt

import torch
from torchvision import transforms
from torchvision.io import read_image
from torch import nn

sys.path.insert(0, "src/open_vocabulary_segmentation")

from models.dinotext import DINOText
from models import build_model


device = "cuda" if torch.cuda.is_available() else "cpu"
cfg_file_path = "src/open_vocabulary_segmentation/configs/cityscapes/dinotext_cityscapes_vitb_mlp_infonce.yml"
# objects = ["table", "bowl", "ramen", "glass", "bottle", "sticks", "chair"]
objects = ["table", "bowl of ramen", "bottle"]

cmap = plt.get_cmap("hsv")

# palette = cmap(np.linspace(0, 1, len(objects) + 1)[:-1])[:, :3]
palette = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])


@torch.no_grad()
def create(model, image_list, data_list, save_folder, text_emb, text):
    model.to(device)
    save_dir = Path(save_folder)
    for image, file_path in zip(image_list, data_list):
        image = image[None].to(device)
        result, _ = model.generate_masks(
            image, img_metas=None, text_emb=text_emb, classnames=text, apply_pamr=True
        )
        result = result.argmax(dim=1).squeeze(0).detach().cpu().numpy()

        img_emb = palette[result] * 2 - 1

        # np.savez_compressed(save_dir / Path(file_path).stem, img_emb)
        np.save(save_dir / Path(file_path).stem, img_emb)


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    seed_num = 42
    seed_everything(seed_num)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)

    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)

    dataset_path = args.dataset_path
    img_folder = os.path.join(dataset_path, "images")
    data_list = os.listdir(img_folder)
    data_list.sort()

    cfg = OmegaConf.load(cfg_file_path)

    model = build_model(cfg.model)
    model.to(device).eval()

    with torch.no_grad():
        text_emb = model.build_dataset_class_tokens("sub_imagenet_template", objects)
        text_emb = model.build_text_embedding(text_emb)

    img_list = []
    for data_path in data_list:
        image_path = os.path.join(img_folder, data_path)
        image = read_image(image_path).float().unsqueeze(0)
        img_list.append(image)

    imgs = torch.cat(img_list)

    save_folder = os.path.join(dataset_path, "talk2dino_features")
    os.makedirs(save_folder, exist_ok=True)

    create(model, imgs, data_list, save_folder, text_emb, objects)
