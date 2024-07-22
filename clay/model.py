import math
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from einops import rearrange, repeat
from huggingface_hub import hf_hub_download
from loguru import logger
from numpy._typing import NDArray
from torch import nn
from tqdm import tqdm
from vit_pytorch.simple_vit import Transformer

from clay.config import config, device
from clay.factory import DynamicEmbedding
from clay.utils import get_datacube, get_stats, posemb_sincos_2d_with_gsd

torch.set_float32_matmul_precision("medium")
os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"


class Encoder(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        mask_ratio=0.0,
        patch_size=8,
        shuffle=False,
        dim=768,
        depth=12,
        heads=12,
        dim_head=64,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.shuffle = shuffle
        self.dim = dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        self.patch_embedding = DynamicEmbedding(
            wave_dim=128,
            num_latent_tokens=128,
            patch_size=patch_size,
            embed_dim=dim,
            is_decoder=False,
        )

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=int(dim * mlp_ratio),
        )

    def to_patch_embed(self, cube, waves):
        """Split the input cube into patches & create embeddings per patch"""
        patches, waves_encoded = self.patch_embedding(cube, waves)  # [B L D]
        return patches, waves_encoded  # ([B L D], [N D])

    def add_encodings(self, patches, time, latlon, gsd):
        """Add position encoding to the patches"""
        B, L, D = patches.shape

        grid_size = int(math.sqrt(L))
        self.num_patches = grid_size**2

        pos_encoding = (
            posemb_sincos_2d_with_gsd(
                h=grid_size,
                w=grid_size,
                dim=(self.dim - 8),
                gsd=gsd,
            )
            .to(patches.device)
            .detach()
        )  # [L (D - 8)]

        time_latlon = torch.hstack((time, latlon)).to(patches.device).detach()  # [B 8]

        pos_encoding = repeat(pos_encoding, "L D -> B L D", B=B)  # [B L (D - 8)]

        time_latlon = repeat(time_latlon, "B D -> B L D", L=L)  # [B L 8]
        pos_metadata_encoding = torch.cat((pos_encoding, time_latlon), dim=-1)  # [B L D]

        patches = patches + pos_metadata_encoding  # [B L D] + [B L D] -> [B L D]
        return patches  # [B L D]

    def mask_out(self, patches):
        """
        Mask out patches randomly by shuffling the patches & masking out the
        first N patches

        Parameters
        ----------
        patches : torch.Tensor A tensor of shape (B, L, D)

        Returns
        -------
        unmasked_patches : torch.Tensor
            A tensor of shape (B, L:(1 - mask_ratio), D) containing the
            embeddings of the unmasked patches.
        unmasked_indices : torch.Tensor
            A tensor of shape (B, (1 - mask_ratio)) containing the indices of
            the unmasked patches.
        masked_indices : torch.Tensor
            A tensor of shape (B, mask_ratio) containing the indices of the
            masked patches.
        masked_matrix : torch.Tensor
            A tensor of shape (B, L) containing the mask matrix, 1 indicates a masked
            patch & 0 indicates an unmasked patch.
        """
        B, L, D = patches.shape
        # assert (
        #     L == self.num_patches
        # ), f"Expected {self.num_patches} patches, got {L} patches."

        if self.shuffle:  # Shuffle the patches
            noise = torch.randn((B, L), device=patches.device)  # [B L]
        else:  # Don't shuffle, useful for interpolation & inspection of embeddings
            noise = rearrange(torch.arange(B * L, device=patches.device), "(B L) -> B L", B=B, L=L)

        random_indices = torch.argsort(noise, dim=-1)  # [B L]
        reverse_indices = torch.argsort(random_indices, dim=-1)  # [B L]

        num_masked_patches = int(self.mask_ratio * self.num_patches)  # Number of patches to be masked out
        masked_indices, unmasked_indices = (
            random_indices[:, :num_masked_patches],  # [B mask_ratio * L]
            random_indices[:, num_masked_patches:],  # [B (1 - mask_ratio) * L]
        )

        # create a mask of shape B L, where 1 indicates a masked patch
        # and 0 indicates an unmasked patch
        masked_matrix = torch.zeros((B, L), device=patches.device)  # [B L] = 0
        masked_matrix[:, :num_masked_patches] = 1  # [B mask_ratio * L] = 1
        masked_matrix = torch.gather(
            masked_matrix, dim=1, index=reverse_indices
        )  # [B L] -> [B L] - reorder the patches

        # mask out the patches
        batch_indices = rearrange(torch.arange(B, device=patches.device), "B -> B 1")  # [B 1]
        unmasked_patches = patches[batch_indices, unmasked_indices, :]  # [B L:(1 - mask_ratio) D]
        _ = patches[batch_indices, masked_indices, :]  # [B L:mask_ratio D]

        return (
            unmasked_patches,
            unmasked_indices,
            masked_indices,
            masked_matrix,
        )  # [B L:(1 - mask_ratio) D], [(1-mask_ratio)], [mask_ratio], [B L]

    def forward(self, datacube):
        cube, time, latlon, gsd, waves = (
            datacube["pixels"],  # [B C H W]
            datacube["time"],  # [B 2]
            datacube["latlon"],  # [B 2]
            datacube["gsd"],  # 1
            datacube["waves"],  # [N]
        )  # [B C H W]

        B, C, H, W = cube.shape

        patches, waves_encoded = self.to_patch_embed(cube, waves)  # [B L D] - patchify & create embeddings per patch

        patches = self.add_encodings(patches, time, latlon, gsd)  # [B L D] - add position encoding to the embeddings

        # mask out patches
        (unmasked_patches, unmasked_indices, masked_indices, masked_matrix) = self.mask_out(
            patches
        )  # [B L:(1 - mask_ratio) D], [(1-mask_ratio)], [mask_ratio], [B L]

        # Add class tokens
        cls_tokens = repeat(self.cls_token, "1 1 D -> B 1 D", B=B)  # [B 1 D]
        unmasked_patches = torch.cat((cls_tokens, unmasked_patches), dim=1)  # [B (1 + L) D]

        # pass the unmasked patches through the transformer
        encoded_unmasked_patches = self.transformer(unmasked_patches)  # [B ((1 + L)):(1 - mask_ratio)) D]

        return (
            encoded_unmasked_patches,
            unmasked_indices,
            masked_indices,
            masked_matrix,
        )  # [B ((1 + L):(1 - mask_ratio)) D], [(1-mask_ratio)], [mask_ratio], [B L]

    def extract_features(self, datacube):
        cube, time, latlon, gsd, waves = (
            datacube["pixels"],  # [B C H W]
            datacube["time"],  # [B 2]
            datacube["latlon"],  # [B 2]
            datacube["gsd"],  # 1
            datacube["waves"],  # [N]
        )

        B, C, H, W = cube.shape

        patches, waves_encoded = self.to_patch_embed(cube, waves)  # [B L D]
        patches = self.add_encodings(patches, time, latlon, gsd)  # [B L D]

        # Add class tokens
        cls_tokens = repeat(self.cls_token, "1 1 D -> B 1 D", B=B)  # [B 1 D]
        patches = torch.cat((cls_tokens, patches), dim=1)  # [B (1 + L) D]

        features = []
        for idx, (attn, ff) in enumerate(self.transformer.layers):
            patches = attn(patches) + patches
            patches = ff(patches) + patches
            _cube = rearrange(patches[:, 1:, :], "B (H W) D -> B D H W", H=H // self.patch_size, W=W // self.patch_size)
            features.append(_cube)

        patches = self.transformer.norm(patches)
        _cube = rearrange(patches[:, 1:, :], "B (H W) D -> B D H W", H=H // self.patch_size, W=W // self.patch_size)
        features.append(_cube)

        return features


class SegmentEncoder(Encoder):
    """
    Encoder class for segmentation tasks, incorporating a feature pyramid
    network (FPN).

    Attributes:
        feature_maps (list): Indices of layers to be used for generating
        feature maps.
        ckpt_path (str): Path to the clay checkpoint file.
    """

    def __init__(  # noqa: PLR0913
        self,
        mask_ratio,
        patch_size,
        shuffle,
        dim,
        depth,
        heads,
        dim_head,
        mlp_ratio,
        feature_maps,
        ckpt_path=None,
    ):
        super().__init__(
            mask_ratio,
            patch_size,
            shuffle,
            dim,
            depth,
            heads,
            dim_head,
            mlp_ratio,
        )
        self.feature_maps = feature_maps

        # Define Feature Pyramid Network (FPN) layers
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
        )

        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
        )

        self.fpn3 = nn.Identity()

        self.fpn4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fpn5 = nn.Identity()

        # Set device
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # Load model from checkpoint if provided

        ckpt_path = hf_hub_download(repo_id=config.hub_repo_id, filename=config.hub_filename)
        self.load_from_ckpt(ckpt_path)

    def load_from_ckpt(self, ckpt_path):
        """
        Load the model's state from a checkpoint file.

        Args:
            ckpt_path (str): The path to the checkpoint file.
        """
        if ckpt_path:
            # Load checkpoint
            ckpt = torch.load(ckpt_path, map_location=self.device)
            state_dict = ckpt.get("state_dict")

            # Prepare new state dict with the desired subset and naming
            new_state_dict = {
                re.sub(r"^model\.encoder\.", "", name): param
                for name, param in state_dict.items()
                if name.startswith("model.encoder")
            }

            # Load the modified state dict into the model
            model_state_dict = self.state_dict()
            for name, param in new_state_dict.items():
                if name in model_state_dict and param.size() == model_state_dict[name].size():
                    model_state_dict[name].copy_(param)
                else:
                    print(f"No matching parameter for {name} with size {param.size()}")

            # Freeze the loaded parameters
            for name, param in self.named_parameters():
                if name in new_state_dict:
                    param.requires_grad = False

    def forward(self, datacube):
        """
        Forward pass of the SegmentEncoder.

        Args:
            datacube (dict): A dictionary containing the input datacube and
                meta information like time, latlon, gsd & wavelenths.

        Returns:
            list: A list of feature maps extracted from the datacube.
        """
        cube, time, latlon, gsd, waves = (
            datacube["pixels"],  # [B C H W]
            datacube["time"],  # [B 2]
            datacube["latlon"],  # [B 2]
            datacube["gsd"],  # 1
            datacube["waves"],  # [N]
        )

        B, C, H, W = cube.shape

        # Patchify and create embeddings per patch
        patches, waves_encoded = self.to_patch_embed(cube, waves)  # [B L D]
        patches = self.add_encodings(patches, time, latlon, gsd)  # [B L D]

        # Add class tokens
        cls_tokens = repeat(self.cls_token, "1 1 D -> B 1 D", B=B)  # [B 1 D]
        patches = torch.cat((cls_tokens, patches), dim=1)  # [B (1 + L) D]

        features = []
        for idx, (attn, ff) in enumerate(self.transformer.layers):
            patches = attn(patches) + patches
            patches = ff(patches) + patches
            if idx in self.feature_maps:
                _cube = rearrange(patches[:, 1:, :], "B (H W) D -> B D H W", H=H // 8, W=W // 8)
                features.append(_cube)
        patches = self.transformer.norm(patches)
        _cube = rearrange(patches[:, 1:, :], "B (H W) D -> B D H W", H=H // 8, W=W // 8)
        features.append(_cube)

        # Apply FPN layers
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4, self.fpn5]
        for i in range(len(features)):
            features[i] = ops[i](features[i])

        return features


class Segmentor(nn.Module):
    """
    Clay Segmentor class that combines the Encoder with FPN layers for semantic
    segmentation.

    Attributes:
        num_classes (int): Number of output classes for segmentation.
        feature_maps (list): Indices of layers to be used for generating feature maps.
        ckpt_path (str): Path to the checkpoint file.
    """

    def __init__(self, num_classes, feature_maps, ckpt_path):
        super().__init__()
        # Default values are for the clay mae base model.
        self.encoder = SegmentEncoder(
            mask_ratio=0.0,
            patch_size=8,
            shuffle=False,
            dim=768,
            depth=12,
            heads=12,
            dim_head=64,
            mlp_ratio=4.0,
            feature_maps=feature_maps,
            ckpt_path=ckpt_path,
        )
        self.upsamples = [nn.Upsample(scale_factor=2**i) for i in range(4)] + [
            nn.Upsample(scale_factor=4),
        ]
        self.fusion = nn.Conv2d(self.encoder.dim * 5, self.encoder.dim, kernel_size=1)
        self.seg_head = nn.Conv2d(self.encoder.dim, num_classes, kernel_size=1)

    def forward(self, datacube):
        """
        Forward pass of the Segmentor.

        Args:
            datacube (dict): A dictionary containing the input datacube and
                meta information like time, latlon, gsd & wavelenths.

        Returns:
            torch.Tensor: The segmentation logits.
        """
        features = self.encoder(datacube)
        for i in range(len(features)):
            features[i] = self.upsamples[i](features[i])

        fused = torch.cat(features, dim=1)
        fused = self.fusion(fused)

        logits = self.seg_head(fused)
        return logits


def get_encoder() -> Encoder:

    ckpt_path = hf_hub_download(repo_id=config.hub_repo_id, filename=config.hub_filename)
    encoder = Encoder()

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict")

    # Remove model.encoder prefix for the clay encoder
    state_dict = {
        re.sub(r"^model\.encoder\.", "", name): param
        for name, param in state_dict.items()
        if name.startswith("model.encoder")
    }

    # Copy the weights from the state dict to the encoder
    for name, param in encoder.named_parameters():
        if name in state_dict and param.size() == state_dict[name].size():
            param.data.copy_(state_dict[name])  # Copy the weights
        else:
            logger.warning(f"No matching parameter for {name} with size {param.size()}")

    # Freeze clay encoder
    for param in encoder.parameters():
        param.requires_grad = False

    # Set the encoder to evaluation mode
    encoder.eval()
    encoder = encoder.to(device)
    return encoder


def get_embeddings_img(
    gsd: float,
    bands: List[str],
    pixels: List[List[List[List[float]]]],  # [B, C, H, W]
    platform: str | None,
    wavelengths: List[float] | None,
    points: List[Tuple[float, float] | None],
    datetimes: List[datetime | None],
) -> NDArray:

    logger.debug(f"Running model inference on {len(points)} samples with batch size {config.batch_size}...")
    stats = get_stats(bands=bands, pixels=pixels, platform=platform, wavelengths=wavelengths)

    embeddings = []
    for i in tqdm(range(0, len(points), config.batch_size)):
        batch_points = points[i : i + config.batch_size]
        batch_pixels = pixels[i : i + config.batch_size]
        batch_datetimes = datetimes[i : i + config.batch_size]

        batch_datacube = get_datacube(
            gsd=gsd, stats=stats, pixels=batch_pixels, datetimes=batch_datetimes, points=batch_points
        )

        with torch.no_grad():
            unmsk_patch, unmsk_idx, msk_idx, msk_matrix = encoder(batch_datacube)
        batch_embedding = unmsk_patch[:, 0, :].cpu().numpy()
        embeddings.append(batch_embedding)

    embedding = np.concatenate(embeddings, axis=0)
    logger.debug("Done!")
    return embedding


def get_embedding(datacube: Dict[str, Any]) -> NDArray:

    with torch.no_grad():
        unmsk_patch, unmsk_idx, msk_idx, msk_matrix = encoder(datacube)
    embedding = unmsk_patch[:, 0, :].cpu().numpy().squeeze()
    logger.info(embedding.shape)
    return embedding


encoder = get_encoder()
