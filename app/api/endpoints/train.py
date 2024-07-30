import asyncio
import json
import pickle
import secrets
import string
from datetime import datetime, timezone
from multiprocessing import Process
from pathlib import Path
from typing import Any, Dict, List, Tuple

import lightning as L
import numpy as np
from fastapi import APIRouter, HTTPException
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.plugins import AsyncCheckpointIO
from loguru import logger

from app.api.endpoints.embeddings import get_embeddings_with_images
from app.config import config, supabase
from app.schemas.clay import ClassificationTrainingDataSample, Image, Images
from app.schemas.train import TrainingJob, TrainingJobRequest
from app.utils.logging import LoggingRoute
from app.utils.misc import random_string
from app.utils.requests import download_file_in_chunks
from app.utils.validation import validate_jsonl
from clay.train import SegmentationDataModule, SegmentorTraining, train_classification

router = APIRouter(route_class=LoggingRoute)


@router.post("")
async def create_training_job(training_job_request: TrainingJobRequest) -> TrainingJob:

    created_at = datetime.now(timezone.utc)
    training_job: Dict[str, Any] = {
        "id": f"trainingjob-{random_string()}",
        "created_at": created_at.isoformat(),
        "status": "initializing",
    } | training_job_request.dict()

    supabase.table(config.SUPABASE_TRAINING_JOBS_TABLE).insert(training_job).execute()

    training_job["created_at"] = int(created_at.timestamp())

    process = Process(target=run_training_job, args=(training_job["id"],))
    process.start()

    return TrainingJob(**training_job)


@router.get("/{training_job_id}")
async def retrieve_training_job(training_job_id: str) -> TrainingJob:
    result = supabase.table(config.SUPABASE_TRAINING_JOBS_TABLE).select("*").eq("id", training_job_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Training job not found")

    training_job = result.data[0]
    training_job["created_at"] = int(datetime.fromisoformat(training_job["created_at"]).timestamp())
    training_job["finished_at"] = (
        int(datetime.fromisoformat(training_job["finished_at"]).timestamp())
        if training_job.get("finished_at")
        else None
    )

    return TrainingJob(**training_job)


@router.delete("/{training_job_id}")
async def cancel_training_job(training_job_id: str) -> TrainingJob:
    result = supabase.table(config.SUPABASE_TRAINING_JOBS_TABLE).select("*").eq("id", training_job_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Training job not found")

    training_job = result.data[0]

    if training_job["status"] in ["succeeded", "failed", "cancelled"]:
        raise HTTPException(
            status_code=400, detail=f"Training job is already finished with status: {training_job['status']}"
        )

    update = {
        "status": "cancelled",
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }

    supabase.table(config.SUPABASE_TRAINING_JOBS_TABLE).update(update).eq("id", training_job_id).execute()

    training_job.update(update)
    training_job["created_at"] = int(datetime.fromisoformat(training_job["created_at"]).timestamp())
    training_job["finished_at"] = int(datetime.fromisoformat(training_job["finished_at"]).timestamp())

    return TrainingJob(**training_job)


def run_training_job(training_job_id: str):
    asyncio.run(execute_training_job(training_job_id))


async def execute_training_job(training_job_id: str):

    training_job = (
        supabase.table(config.SUPABASE_TRAINING_JOBS_TABLE).select("*").eq("id", training_job_id).execute().data[0]
    )
    logger.info(f"Training job {training_job_id} started")

    try:

        supabase.table(config.SUPABASE_TRAINING_JOBS_TABLE).update({"status": "downloading_files"}).eq(
            "id", training_job_id
        ).execute()

        training_file_path = config.FILES_CACHE_DIR / training_job["training_file"]
        if not training_file_path.exists():
            download_url = supabase.storage.from_(config.SUPABASE_FILES_BUCKET).get_public_url(
                training_job["training_file"]
            )
            await download_file_in_chunks(url=download_url, path=training_file_path)

        validation_file_path = None
        if training_job["validation_file"]:
            validation_file_path = config.FILES_CACHE_DIR / training_job["validation_file"]
            if not validation_file_path.exists():
                download_url = supabase.storage.from_(config.SUPABASE_FILES_BUCKET).get_public_url(
                    training_job["validation_file"]
                )
                await download_file_in_chunks(url=download_url, path=validation_file_path)

        supabase.table(config.SUPABASE_TRAINING_JOBS_TABLE).update({"status": "validating_files"}).eq(
            "id", training_job_id
        ).execute()

        validate_jsonl(file_path=training_file_path, task=training_job["task"])
        if validation_file_path:
            validate_jsonl(file_path=validation_file_path, task=training_job["task"])

        supabase.table(config.SUPABASE_TRAINING_JOBS_TABLE).update({"status": "running"}).eq(
            "id", training_job_id
        ).execute()

        model_name = ""
        if training_job["task"] == "classification":
            model_name = await train_classification_model(training_file_path, validation_file_path)

        supabase.table(config.SUPABASE_TRAINING_JOBS_TABLE).update(
            {
                "status": "succeeded",
                "trained_model": model_name,
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
        ).eq("id", training_job_id).execute()

    except Exception as e:
        supabase.table(config.SUPABASE_TRAINING_JOBS_TABLE).update(
            {
                "status": "failed",
                "error": f"{e.__class__.__name__}: {str(e)}",
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
        ).eq("id", training_job_id).execute()


async def train_classification_model(
    training_file_path: Path,
    validation_file_path: Path | None = None,
    hyperparameters: Dict[str, Any] | None = None,
) -> str:

    training_images: List[Image] = []
    training_labels: List[int] = []

    with open(training_file_path, "r") as file:
        for line in file:
            data = json.loads(line)
            sample = ClassificationTrainingDataSample(**data)
            training_images.append(sample.image)
            training_labels.append(sample.label)

    embeddings = await get_embeddings_with_images(images=Images(images=training_images))

    model, _ = train_classification(embeddings=np.array(embeddings.embeddings), labels=np.array(training_labels))

    model_name = f"model:classification-{random_string()}"
    model_path = config.FILES_CACHE_DIR / model_name
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    supabase.storage.from_(config.SUPABASE_MODEL_BUCKET).upload(path=model_name, file=model_path)

    return model_name


async def train_classification_model(data: TrainClassificationData) -> TrainResults:
    """Train classification model on your data."""
    embeddings = await get_embeddings_with_images(data)
    model, details = train_classification(embeddings=np.array(embeddings.embeddings), labels=np.array(data.labels))
    model_bytes = pickle.dumps(model)
    model_name = f"checkpoints/classification/{len(embeddings.embeddings)}_{''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(5))}.pkl"
    response = supabase.storage.from_(config.SUPABASE_MODEL_BUCKET).upload(path=model_name, file=model_bytes)
    return TrainResults(model=model_name, train_details=details)


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
