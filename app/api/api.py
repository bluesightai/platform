import os
import pickle
import secrets
import string
from datetime import datetime
from typing import List, Tuple

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import APIRouter, Depends, HTTPException, status
from gotrue import UserResponse
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.plugins import AsyncCheckpointIO
from loguru import logger

from app.config import config, supabase
from app.schemas.auth import LoginData, Token, UserData
from app.schemas.clay import (ClassificationLabels, Embeddings, Images, InferenceData, Points, SegmentationLabel,
                              SegmentationLabels, TrainClassificationData, TrainResults, TrainSegmentationData)
from app.schemas.common import TextResponse
from app.utils.auth import get_current_user
from app.utils.logging import LoggingRoute
from clay.model import get_embedding, get_embeddings_img
from clay.train import (SegmentationDataModule, SegmentationDataset, SegmentorTraining, predict_classification,
                        train_classification)
from clay.utils import get_catalog_items, get_stack, stack_to_datacube

api_router = APIRouter(route_class=LoggingRoute)


@api_router.post("/train/classification", tags=["Train"])
async def train_classification_model(data: TrainClassificationData) -> TrainResults:
    """Train classification model on your data."""
    embeddings = await get_embeddings_with_images(data)
    model, details = train_classification(embeddings=np.array(embeddings.embeddings), labels=np.array(data.labels))
    model_bytes = pickle.dumps(model)
    model_name = f"classification_{len(embeddings.embeddings)}_{''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(5))}"
    response = supabase.storage.from_(config.SUPABASE_MODEL_BUCKET).upload(path=model_name + ".pkl", file=model_bytes)
    return TrainResults(model_id=model_name, train_details=details)


@api_router.post("/inference/classification", tags=["Train"])
async def infer_classification_model(data: InferenceData) -> ClassificationLabels:
    """Run inference of previously trained classification model on your data."""
    embeddings = await get_embeddings_with_images(data)
    model = pickle.loads(supabase.storage.from_(config.SUPABASE_MODEL_BUCKET).download(path=data.model_id + ".pkl"))
    labels = predict_classification(clf=model, embeddings=np.array(embeddings.embeddings))
    return ClassificationLabels(labels=labels.tolist())


@api_router.post("/train/segmentation", tags=["Train"])
async def train_segmentation_model(data: TrainSegmentationData) -> TrainResults:
    """Train segmentation model on your data."""

    pixels: List[List[List[List[float]]]] = []
    points: List[Tuple[float, float] | None] = []
    datetimes: List[datetime | None] = []
    # Check consistency of platform, gsd, and bands
    first_image = data.images[0]
    platform, gsd, bands = first_image.platform, first_image.gsd, first_image.bands
    pixel_shape = None
    for image in data.images:
        if image.platform != platform:
            raise ValueError("Inconsistent platform across images")
        if image.gsd != gsd:
            raise ValueError("Inconsistent gsd across images")
        if image.bands != bands:
            raise ValueError("Inconsistent bands across images")

        if pixel_shape is None:
            pixel_shape = len(image.pixels), len(image.pixels[0]), len(image.pixels[0][0])
        elif (len(image.pixels), len(image.pixels[0]), len(image.pixels[0][0])) != pixel_shape:
            raise ValueError("Inconsistent pixel shapes across images")

        pixels.append(image.pixels)
        points.append(image.point)
        datetimes.append(datetime.fromtimestamp(image.timestamp) if image.timestamp else None)
    labels = [label.label for label in data.labels]
    flat_labels = [label for batch in labels for row in batch for label in row]
    unique_classes = np.unique(flat_labels)
    num_classes = len(unique_classes)
    logger.info(f"Found {num_classes} unique classes!")

    data_module = SegmentationDataModule(
        gsd=data.images[0].gsd,
        bands=data.images[0].bands,
        pixels=pixels,
        platform=data.images[0].platform,
        wavelengths=data.images[0].wavelengths,
        points=points,
        datetimes=datetimes,
        labels=labels,
        batch_size=40,
        num_workers=0,
    )

    model = SegmentorTraining(
        num_classes=num_classes,
        feature_maps=[3, 5, 7, 11],
        ckpt_path="checkpoints/clay-v1-base.ckpt",
        lr=1e-5,
        wd=0.05,
        b1=0.9,
        b2=0.95,
    )

    save_path = f"checkpoints/segmentation/{len(pixels)}_{''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(5))}"
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path,
        filename="7class-segment_epoch-{epoch:02d}_val-iou-{val/iou:.4f}",
        monitor="val/iou",
        mode="max",
        save_last=True,
        save_top_k=2,
        save_weights_only=True,
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = L.Trainer(
        accelerator="auto",
        strategy="ddp",
        devices="auto",
        num_nodes=1,
        precision="bf16-mixed",
        log_every_n_steps=5,
        max_epochs=10,
        accumulate_grad_batches=1,
        default_root_dir="checkpoints/segment",
        fast_dev_run=False,
        num_sanity_val_steps=0,
        # logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        plugins=[AsyncCheckpointIO()],
    )

    L.seed_everything(42)

    trainer.fit(model, datamodule=data_module)

    model_id = save_path + ".ckpt"
    with open(save_path + "/last.ckpt", "rb") as f:
        response = supabase.storage.from_(config.SUPABASE_MODEL_BUCKET).upload(path=model_id, file=f)

    return TrainResults(model_id=model_id, train_details=None)


@api_router.post("/inference/segmentation", tags=["Train"])
async def infer_segmentation_model(data: InferenceData) -> SegmentationLabels:
    """Run inference of previously trained segmentation model on your data."""

    checkpoint_download_path = f".cache/{data.model_id}"
    os.makedirs(os.path.dirname(checkpoint_download_path), exist_ok=True)
    with open(checkpoint_download_path, "wb+") as f:
        res = supabase.storage.from_(config.SUPABASE_MODEL_BUCKET).download(data.model_id)
        f.write(res)

    pixels: List[List[List[List[float]]]] = []
    points: List[Tuple[float, float] | None] = []
    datetimes: List[datetime | None] = []
    # Check consistency of platform, gsd, and bands
    first_image = data.images[0]
    platform, gsd, bands = first_image.platform, first_image.gsd, first_image.bands
    pixel_shape = None
    for image in data.images:
        if image.platform != platform:
            raise ValueError("Inconsistent platform across images")
        if image.gsd != gsd:
            raise ValueError("Inconsistent gsd across images")
        if image.bands != bands:
            raise ValueError("Inconsistent bands across images")

        if pixel_shape is None:
            pixel_shape = len(image.pixels), len(image.pixels[0]), len(image.pixels[0][0])
        elif (len(image.pixels), len(image.pixels[0]), len(image.pixels[0][0])) != pixel_shape:
            raise ValueError("Inconsistent pixel shapes across images")

        pixels.append(image.pixels)
        points.append(image.point)
        datetimes.append(datetime.fromtimestamp(image.timestamp) if image.timestamp else None)

    # model = SegmentorTraining.load_from_checkpoint("./checkpoints/segment/last.ckpt")
    model = SegmentorTraining.load_from_checkpoint(checkpoint_download_path)
    data_module = SegmentationDataModule(
        gsd=data.images[0].gsd,
        bands=data.images[0].bands,
        pixels=pixels,
        platform=data.images[0].platform,
        wavelengths=data.images[0].wavelengths,
        points=points,
        datetimes=datetimes,
        batch_size=40,
        num_workers=0,
        train_test_ratio=0,
        labels=None,
    )
    data_module.setup()

    all_preds = []
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        for batch in data_module.val_dataloader():
            outputs = model(batch)
            outputs = F.interpolate(
                outputs,
                size=(batch["pixels"].shape[-2], batch["pixels"].shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            all_preds.append(preds)

    # Combine all predictions
    all_preds = np.concatenate(all_preds, axis=0)

    logger.warning(all_preds.shape)

    return SegmentationLabels(labels=[SegmentationLabel(label=label) for label in all_preds.tolist()[: len(pixels)]])


@api_router.post("/embeddings/img", tags=["Embeddings"])
async def get_embeddings_with_images(images: Images) -> Embeddings:
    """Get embeddings for a list of images."""
    pixels: List[List[List[List[float]]]] = []
    points: List[Tuple[float, float] | None] = []
    datetimes: List[datetime | None] = []
    # Check consistency of platform, gsd, and bands
    first_image = images.images[0]
    platform, gsd, bands = first_image.platform, first_image.gsd, first_image.bands
    pixel_shape = None
    for image in images.images:
        if image.platform != platform:
            raise ValueError("Inconsistent platform across images")
        if image.gsd != gsd:
            raise ValueError("Inconsistent gsd across images")
        if image.bands != bands:
            raise ValueError("Inconsistent bands across images")

        if pixel_shape is None:
            pixel_shape = len(image.pixels), len(image.pixels[0]), len(image.pixels[0][0])
        elif (len(image.pixels), len(image.pixels[0]), len(image.pixels[0][0])) != pixel_shape:
            raise ValueError("Inconsistent pixel shapes across images")

        pixels.append(image.pixels)
        points.append(image.point)
        datetimes.append(datetime.fromtimestamp(image.timestamp) if image.timestamp else None)

    embeddings = get_embeddings_img(
        gsd=images.images[0].gsd,
        bands=images.images[0].bands,
        pixels=pixels,
        platform=images.images[0].platform,
        wavelengths=images.images[0].wavelengths,
        points=points,
        datetimes=datetimes,
    ).tolist()
    return Embeddings(embeddings=embeddings)


@api_router.post("/embeddings/loc", tags=["Embeddings"])
async def get_embeddings_with_coordinates(points: Points) -> Embeddings:
    """Get embeddings for a list of points."""
    items = [get_catalog_items(lat=lat, lon=lon, start="2022-01-01") for lat, lon in points.points]
    stacks = [
        get_stack(lat=lat, lon=lon, items=item, size=points.size, gsd=0.6)
        for item, (lat, lon) in zip(items, points.points)
    ]
    datacubes = [stack_to_datacube(lat=lat, lon=lon, stack=stack) for stack, (lat, lon) in zip(stacks, points.points)]
    return Embeddings(embeddings=[get_embedding(datacube=datacube).tolist() for datacube in datacubes])


@api_router.post("/auth/register", tags=["Authentication"], include_in_schema=False)
async def register(user_data: UserData) -> TextResponse:
    """Login endpoint

    ## This is login endpoint
    """
    try:
        response = supabase.auth.sign_up(
            {
                "email": user_data.email,
                "password": user_data.password,
                "options": {"data": {"first_name": user_data.first_name, "last_name": user_data.last_name}},
            }
        )
        return TextResponse(message="Check your email for verification!")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@api_router.get("/auth/token", response_model=Token, tags=["Authentication"], include_in_schema=False)
async def get_token(login_data: LoginData) -> Token:
    try:
        response = supabase.auth.sign_in_with_password({"email": login_data.email, "password": login_data.password})
        if not response.session:
            raise HTTPException(status_code=400, detail="User not found")
        access_token = response.session.access_token
        return Token(access_token=access_token, token_type="bearer")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@api_router.get("/auth/me", tags=["Authentication"], include_in_schema=False)
async def get_my_data(current_user: UserResponse = Depends(get_current_user)):
    return current_user


# @api_router.post("/inference/classification", tags=["inference"])
# async def inference_classification(model_id: str, images: List[ImageData]) -> List[int]:
#     """Embeddings endpoint"""
#     return [2, 1]
