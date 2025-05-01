#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import sys
from os import makedirs
from argparse import ArgumentParser
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from omegaconf import OmegaConf

import torch
import torchvision
import torch.nn.functional as F

from tqdm import tqdm
from scene import Scene
from gaussian_renderer import render, GaussianModel
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, get_combined_args
from autoencoder.model import Autoencoder


sys.path.insert(0, "src/open_vocabulary_segmentation")

from models.dinotext import DINOText
from models import build_model


device = "cuda" if torch.cuda.is_available() else "cpu"
cfg_file_path = "src/open_vocabulary_segmentation/configs/cityscapes/dinotext_cityscapes_vitb_mlp_infonce.yml"


@torch.no_grad()
def get_text_embedding(model, autoencoder, text):
    text_emb = model.build_dataset_class_tokens("sub_imagenet_template", [text])
    text_emb = model.build_text_embedding(text_emb)

    text_emb = autoencoder.encode(text_emb)

    return text_emb.squeeze(0)


def render_set(
    model_path,
    source_path,
    name,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    args,
    text_emb,
):
    text_emb = text_emb[:, None, None]

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    render_npy_path = os.path.join(
        model_path, name, "ours_{}".format(iteration), "renders_npy"
    )
    gts_npy_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_npy")

    similarity_path = os.path.join(
        model_path, name, "ours_{}".format(iteration), "similarity"
    )  # Новый путь

    makedirs(render_npy_path, exist_ok=True)
    makedirs(gts_npy_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    makedirs(similarity_path, exist_ok=True)  # Создаем папку для карт схожести

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        output = render(view, gaussians, pipeline, background, args)

        if not args.include_feature:
            rendering = output["render"]
        else:
            rendering = output["language_feature_image"]

        if not args.include_feature:
            gt = view.original_image[0:3, :, :]

        else:
            # gt = view.get_language_feature(
            #     os.path.join(source_path, args.language_features_name),
            #     feature_level=args.feature_level,
            # )
            gt = view.get_dino_feature(source_path)

        np.save(
            os.path.join(render_npy_path, "{0:05d}".format(idx) + ".npy"),
            rendering.permute(1, 2, 0).cpu().numpy(),
        )
        np.save(
            os.path.join(gts_npy_path, "{0:05d}".format(idx) + ".npy"),
            gt.permute(1, 2, 0).cpu().numpy(),
        )
        torchvision.utils.save_image(
            rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png")
        )
        torchvision.utils.save_image(
            gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png")
        )


        print("Text emb", text_emb.min(), text_emb.max())
        print("Render", rendering.min(), rendering.max())
        similarities = F.cosine_similarity(text_emb, rendering, dim=0).clamp(0, 1)
        print("Similarity", similarities.min(), similarities.max())

        torchvision.utils.save_image(
            similarities, os.path.join(similarity_path, "{0:05d}".format(idx) + ".png")
        )

        # if text_emb and args.include_feature:
        #     # Получаем DINO features для рендера
        #     dino_features = output["language_feature_image"]  # [C, H, W]

        #     # Нормализуем фичи
        #     dino_features = dino_features.permute(1, 2, 0)  # [H, W, C]
        #     dino_features = dino_features / dino_features.norm(dim=-1, keepdim=True)

        #     # Вычисляем косинусную схожесть
        #     similarity = F.cosine_similarity(
        #         dino_features.reshape(-1, dino_features.shape[-1]),
        #         text_embedding,
        #         dim=-1,
        #     ).reshape(dino_features.shape[:2])

        #     # Нормализуем схожесть для визуализации
        #     similarity = (similarity - similarity.min()) / (
        #         similarity.max() - similarity.min()
        #     )

        #     # Сохраняем карту схожести
        #     plt.imsave(
        #         os.path.join(similarity_path, f"{idx:05d}_similarity.png"),
        #         similarity.cpu().numpy(),
        #         cmap=cm.jet,
        #         vmin=0,
        #         vmax=1,
        #     )

        #     # Сохраняем наложенную визуализацию
        #     overlay = (
        #         similarity.unsqueeze(-1).cpu().numpy()
        #         * cm.jet(similarity.cpu().numpy())[..., :3]
        #         * 255
        #     ).astype(np.uint8)
        #     overlay_img = Image.fromarray(overlay)
        #     original_img = torchvision.transforms.ToPILImage()(rendering.cpu())
        #     original_img = original_img.convert("RGBA")
        #     overlay_img = overlay_img.convert("RGBA")
        #     blended = Image.blend(original_img, overlay_img, alpha=0.5)
        #     blended.save(os.path.join(similarity_path, f"{idx:05d}_overlay.png"))


def render_sets(
    dataset: ModelParams,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    args,
    text_request=None,  # текстовый запрос
):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, shuffle=False)
        checkpoint = os.path.join(args.model_path, "chkpnt30000.pth")
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, args, mode="test")

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(
                dataset.model_path,
                dataset.source_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
                args,
                text_request,  # Текстовый запрос
            )

        if not skip_test:
            render_set(
                dataset.model_path,
                dataset.source_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
                args,
                text_request,  # Текстовый запрос
            )


if __name__ == "__main__":
    # Set up command line argument parser

    parser = ArgumentParser(description="Testing script parameters")
    model_params = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--include_feature", action="store_true")
    parser.add_argument("--text_request", type=str, default=None)
    parser.add_argument("--autoencoder_checkpoint", type=str)
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

    cfg = OmegaConf.load(cfg_file_path)

    model = build_model(cfg.model)
    model.to(device).eval()

    args = get_combined_args(parser)

    autoencoder = Autoencoder(
        encoder_hidden_dims=args.encoder_dims, decoder_hidden_dims=args.decoder_dims
    )
    autoencoder.to(device).eval()
    autoencoder.load_state_dict(
        torch.load(args.autoencoder_checkpoint, weights_only=True)
    )

    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    text_emb = get_text_embedding(model, autoencoder, args.text_request)
    print(text_emb)
    render_sets(
        model_params.extract(args),
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args,
        text_emb,
    )
