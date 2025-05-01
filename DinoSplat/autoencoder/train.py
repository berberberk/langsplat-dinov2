import os
from statistics import mean

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm
from dataset import AutoencoderDataset
from model import Autoencoder

import argparse


torch.autograd.set_detect_anomaly(True)


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def cos_loss(network_output, gt):
    return 1 - F.cosine_similarity(network_output, gt, dim=0).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
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
    parser.add_argument("--dataset_name", type=str, required=True)
    args = parser.parse_args()
    dataset_path = args.dataset_path
    num_epochs = args.num_epochs
    data_dir = f"{dataset_path}/dino_features"
    os.makedirs(f"ckpt/{args.dataset_name}", exist_ok=True)
    train_dataset = AutoencoderDataset(data_dir)
    train_dataset, test_dataset = torch.utils.data.random_split(
        train_dataset, [0.2, 0.8]
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1024,
        shuffle=True,
        num_workers=64,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=64,
        drop_last=False,
    )

    encoder_hidden_dims = args.encoder_dims
    decoder_hidden_dims = args.decoder_dims

    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_eval_loss = float("inf")
    best_epoch = 0

    loop = tqdm(range(num_epochs))
    for epoch in loop:
        train_losses = []
        test_losses = []
        model.train()

        for idx, feature in enumerate(train_loader):
            data = feature.to("cuda:0")
            outputs_dim3 = model.encode(data)
            outputs = model.decode(outputs_dim3)

            l2loss = l2_loss(outputs, data)
            cosloss = cos_loss(outputs, data)

            loss = l2loss + cosloss * 0.001

            train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        for feature in test_loader:
            data = feature.to("cuda:0")
            with torch.no_grad():
                outputs = model(data)
            loss = l2_loss(outputs, data) + cos_loss(outputs, data) * 0.001
            test_losses.append(loss.item())

        loop.set_postfix(
            {
                "train_loss": mean(train_losses),
                "test_loss": mean(test_losses),
            }
        )
        if mean(test_losses) < best_eval_loss:
            best_eval_loss = mean(test_losses)
            best_epoch = epoch
            torch.save(model.state_dict(), f"ckpt/{args.dataset_name}/best_ckpt.pth")

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"ckpt/{args.dataset_name}/{epoch}_ckpt.pth")

    print(f"best_epoch: {best_epoch}")
    print("best_loss: {:.8f}".format(best_eval_loss))


if __name__ == "__main__":
    main()
