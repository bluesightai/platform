import asyncio
import pickle
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import lightning as L
import numpy as np
from fastapi import APIRouter, HTTPException
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.plugins import AsyncCheckpointIO
from loguru import logger
from supabase.client import AsyncClient

from app.api.deps import SessionDep
from app.api.endpoints.models import upload_model
from app.config import config
from app.crud.training_job import crud_training_job
from app.schemas.training_job import TrainingJob, TrainingJobCreate, TrainingJobUpdate
from app.utils.logging import LoggingRoute
from app.utils.requests import download_file_from_bucket
from app.utils.validation import validate_hdf5
from clay.model import get_embeddings_img
from clay.train import SegmentationDataModule, SegmentorTraining, train_classification

router = APIRouter(route_class=LoggingRoute)


@router.post("")
async def create_training_job(training_job_create: TrainingJobCreate, session: SessionDep) -> TrainingJob:
    training_job = await crud_training_job.create(db=session, obj_in=training_job_create)
    asyncio.create_task(execute_training_job(training_job, session))
    return training_job


@router.get("/{training_job_id}")
async def retrieve_training_job(training_job_id: str, session: SessionDep) -> TrainingJob:
    training_job = await crud_training_job.get(db=session, id=training_job_id)
    if not training_job:
        raise HTTPException(status_code=404, detail="Training job not found")
    return training_job


@router.delete("/{training_job_id}")
async def cancel_training_job(training_job_id: str, session: SessionDep) -> TrainingJob:
    training_job = await retrieve_training_job(training_job_id, session)
    if training_job.status in ["succeeded", "failed", "cancelled"]:
        raise HTTPException(
            status_code=400, detail=f"Training job is already finished with status: {training_job.status}"
        )
    training_job = await crud_training_job.update(
        db=session,
        id=training_job_id,
        obj_in=TrainingJobUpdate(status="cancelled", finished_at=datetime.now(timezone.utc).timestamp()),
    )
    return training_job


async def execute_training_job(training_job: TrainingJob, session: AsyncClient):
    logger.debug(f"Starting training job {training_job.id}")
    try:

        logger.debug(f"Downloading files for training job {training_job.id}")
        await crud_training_job.update(
            db=session, id=training_job.id, obj_in=TrainingJobUpdate(status="downloading_files")
        )
        training_file_path = await download_file_from_bucket(session=session, file_id=training_job.training_file)
        validation_file_path = None
        if training_job.validation_file:
            validation_file_path = await download_file_from_bucket(
                session=session, file_id=training_job.validation_file
            )

        logger.debug(f"Validating training file {training_file_path} and validation file {validation_file_path}")
        await crud_training_job.update(
            db=session, id=training_job.id, obj_in=TrainingJobUpdate(status="validating_files")
        )
        validate_hdf5(file_path=training_file_path, task=training_job.task)
        if validation_file_path:
            validate_hdf5(file_path=validation_file_path, task=training_job.task)

        logger.debug(f"Training job {training_job.id} is running!")
        await crud_training_job.update(db=session, id=training_job.id, obj_in=TrainingJobUpdate(status="running"))
        local_model_path = Path()
        if training_job.task == "classification":
            local_model_path = await train_classification_model(training_file_path, validation_file_path)
        elif training_job.task == "segmentation":
            local_model_path = await train_segmentation_model(training_file_path, validation_file_path)

        logger.debug(f"Uploading model for training job {training_job.id} to storage")
        model_metadata = await upload_model(session=session, task=training_job.task, local_model_path=local_model_path)

        logger.debug(f"Training job {training_job.id} succeeded! Model ID: {model_metadata.id}")
        await crud_training_job.update(
            db=session,
            id=training_job.id,
            obj_in=TrainingJobUpdate(
                status="succeeded",
                trained_model=model_metadata.id,
                finished_at=datetime.now(timezone.utc).timestamp(),
            ),
        )

    except Exception as e:
        logger.error(f"Training job {training_job.id} failed with error: {e.__class__.__name__}: {str(e)}")
        await crud_training_job.update(
            db=session,
            id=training_job.id,
            obj_in=TrainingJobUpdate(
                status="failed",
                error=f"{e.__class__.__name__}: {str(e)}",
                finished_at=datetime.now(timezone.utc).timestamp(),
            ),
        )


async def train_classification_model(
    training_file_path: Path,
    validation_file_path: Path | None = None,
    hyperparameters: Dict[str, Any] | None = None,
) -> Path:

    pixels: List[List[List[List[float]]]] = []
    points: List[Tuple[float, float] | None] = []
    datetimes: List[datetime | None] = []
    training_labels: List[int] = []
    with h5py.File(training_file_path, "r") as f:
        first_item: h5py.Dataset = f["data"][0]
        gsd: float = first_item["gsd"].item()
        bands: list[str] = [band.decode("ascii") for band in first_item["bands"]]
        platform: str | None = first_item["platform"].decode("ascii") or None

        for sample in f["data"]:

            pixels.append(sample["pixels"])
            points.append(sample["point"].tolist() if sum(sample["point"]) != 0 else None)
            datetimes.append(datetime.fromtimestamp(sample["timestamp"].item()) if sample["timestamp"] else None)
            training_labels.append(sample["label"].item())

    embeddings = get_embeddings_img(
        gsd=gsd,
        bands=bands,
        pixels=pixels,
        platform=platform,
        wavelengths=None,
        points=points,
        datetimes=datetimes,
    )

    model, _ = train_classification(embeddings=embeddings, labels=np.array(training_labels))

    local_model_path = config.CACHE_DIR / uuid.uuid4().hex
    with open(local_model_path, "wb") as f:
        pickle.dump(model, f)

    return local_model_path


async def train_segmentation_model(
    training_file_path: Path,
    validation_file_path: Path | None = None,
    hyperparameters: Dict[str, Any] | None = None,
) -> Path:

    # num_classes = len(np.unique(data.labels))
    num_classes = 7
    logger.info(f"Found {num_classes} unique classes!")

    data_module = SegmentationDataModule(
        train_file_path=training_file_path,
        validation_file_path=validation_file_path,
        batch_size=20,
        num_workers=0,
    )

    model = SegmentorTraining(
        num_classes=num_classes,
        feature_maps=[3, 5, 7, 11],
        lr=1e-5,
        wd=0.05,
        b1=0.9,
        b2=0.95,
    )

    checkpoint_folder_path = config.CHECKPOINTS_DIR / uuid.uuid4().hex
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_folder_path,
        filename=str(num_classes) + "class-segment_epoch-{epoch:02d}_val-iou-{(val/iou if val/iou else train/iou):.4f}",
        monitor="val/iou" if validation_file_path else "train/iou",
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
        # default_root_dir="checkpoints/segment",
        fast_dev_run=False,
        num_sanity_val_steps=0,
        # logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        plugins=[AsyncCheckpointIO()],
        val_check_interval=0 if not validation_file_path else None,
        limit_val_batches=0 if not validation_file_path else None,
    )

    L.seed_everything(42)

    trainer.fit(model, datamodule=data_module)

    local_model_path = checkpoint_folder_path / "last.ckpt"

    return local_model_path
