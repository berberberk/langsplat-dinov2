from pathlib import Path

import numpy as np
import cv2
import torch
from torchvision.transforms import functional as VF

import argparse

from dataset import AutoencoderDataset
from model import Autoencoder

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument(
        "--encoder_dims",
        nargs="+",
        type=int,
        default=[512, 256, 128, 64, 32, 3],
    )
    parser.add_argument(
        "--decoder_dims",
        nargs="+",
        type=int,
        default=[16, 32, 64, 128, 256, 512, 768],
    )

    args = parser.parse_args()

    dataset_name = Path(args.dataset_name)
    encoder_hidden_dims = args.encoder_dims
    decoder_hidden_dims = args.decoder_dims
    dataset_path = args.dataset_path
    ckpt_path = f"ckpt/{dataset_name}/best_ckpt.pth"

    data_dir = f"{dataset_path}/dino_features"
    images_dir = Path(f"{dataset_path}/images")
    output_dir = Path(f"{dataset_path}/dino_features_dim3")

    random_image_path = next(images_dir.iterdir())
    mat = cv2.imread(str(random_image_path))
    orig_h, orig_w, _ = mat.shape

    checkpoint = torch.load(ckpt_path)
    train_dataset = AutoencoderDataset(data_dir)

    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")

    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)

    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for path, feat in train_dataset.data_dic.items():
            feat = torch.tensor(feat).to(device)
            feat = feat.view(1, 37, 37, -1).permute(0, 3, 1, 2)
            feat = VF.resize(feat, (orig_h, orig_w)).permute(0, 2, 3, 1)
            output = (
                model.encode(torch.tensor(feat)[None].to(device))
                .view(orig_h, orig_w, 3)
                .detach()
                .cpu()
                .numpy()
            )
            save_path = output_dir / Path(path).name
            np.savez_compressed(save_path, output)


if __name__ == "__main__":
    main()
