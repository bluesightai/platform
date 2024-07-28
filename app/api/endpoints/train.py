import pickle
import secrets
import string
from datetime import datetime
from typing import List, Tuple

import lightning as L
import numpy as np
from fastapi import APIRouter
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.plugins import AsyncCheckpointIO
from loguru import logger

from app.api.endpoints.embeddings import get_embeddings_with_images
from app.config import config, supabase
from app.schemas.clay import TrainClassificationData, TrainResults, TrainSegmentationData
from clay.train import SegmentationDataModule, SegmentorTraining, train_classification

router = APIRouter()


@router.post("/classification")
async def train_classification_model(data: TrainClassificationData) -> TrainResults:
    """Train classification model on your data."""
    embeddings = await get_embeddings_with_images(data)
    model, details = train_classification(embeddings=np.array(embeddings.embeddings), labels=np.array(data.labels))
    model_bytes = pickle.dumps(model)
    model_name = f"checkpoints/classification/{len(embeddings.embeddings)}_{''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(5))}.pkl"
    response = supabase.storage.from_(config.SUPABASE_MODEL_BUCKET).upload(path=model_name, file=model_bytes)
    return TrainResults(model=model_name, train_details=details)


@router.post("/segmentation")
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
    unique_classes = np.unique(data.labels)
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
        labels=data.labels,
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

    return TrainResults(model=model_id, train_details=None)
