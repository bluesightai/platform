import numpy as np
import open_clip
import torch
from huggingface_hub import hf_hub_download
from loguru import logger
from numpy.typing import NDArray
from PIL import Image
from torchvision.transforms.transforms import Compose

from clay.config import config, device

ckpt_path = hf_hub_download(repo_id=config.clip_hub_repo_id, filename=config.clip_hub_filename)

model, _, preprocess = open_clip.create_model_and_transforms(
    model_name="ViT-L-14", pretrained=ckpt_path, force_quick_gelu=True, device=device
)
if not isinstance(preprocess, Compose):
    raise ValueError("Expected preprocess to be a Compose object")

model.eval()
tokenizer = open_clip.get_tokenizer("ViT-L-14")


def get_embeddings_from_images(images: list[NDArray[np.uint8]]) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for i in range(0, len(images), config.batch_size):
        batch = images[i : i + config.batch_size]
        images_transformed: list[torch.Tensor] = [preprocess(Image.fromarray(image)) for image in batch]
        inputs = torch.stack(images_transformed).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            batch_embeddings = model.encode_image(inputs)
            batch_embeddings /= batch_embeddings.norm(dim=-1, keepdim=True)

        embeddings.extend(batch_embeddings.cpu().tolist())
    return embeddings


def get_embeddings_from_text(texts: list[str]) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for i in range(0, len(texts), config.batch_size):
        batch = texts[i : i + config.batch_size]
        inputs = tokenizer(batch).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            batch_embeddings = model.encode_text(inputs)
            batch_embeddings /= batch_embeddings.norm(dim=-1, keepdim=True)
        embeddings.extend(batch_embeddings.cpu().tolist())
    return embeddings
