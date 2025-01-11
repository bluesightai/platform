from copy import deepcopy
import pickle
import io
import base64
from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import APIRouter
from loguru import logger
from pandas._libs.tslibs.timezones import infer_tzinfo

from app.api.deps import SessionDep
from app.api.endpoints.embeddings import get_embeddings_with_images
from app.api.endpoints.models import download_model
from app.config import config
from app.schemas.clay import ClassificationLabels, EmbeddingsRequestBase, Images, InferenceData, SegmentationLabels
from app.utils.logging import LoggingRoute
from clay.train import SegmentationDataModuleInference, SegmentorTraining, predict_classification

router = APIRouter(route_class=LoggingRoute)


@router.post("")
async def run_trained_model_inference(
    data: InferenceData, session: SessionDep
) -> SegmentationLabels | ClassificationLabels:
    """Run inference of trained model on your data."""

    model_metadata, local_model_path = await download_model(session, data.model)

    if model_metadata.task == "classification":
        with open(local_model_path, "rb") as f:
            model = pickle.load(f)

        processed_images = []
        for img_dict in data.images:
            img = deepcopy(img_dict.model_dump())
            buffer = io.BytesIO()
            np.save(buffer, np.array(img["pixels"]))
            img["pixels"] = base64.b64encode(buffer.getvalue()).decode("utf-8")
            processed_images.append(img)
        embeddings = await get_embeddings_with_images(EmbeddingsRequestBase(images=processed_images), session)
        labels = predict_classification(clf=model, embeddings=np.array(embeddings.embeddings))
        return ClassificationLabels(labels=labels.tolist())
    elif model_metadata.task == "segmentation":

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

            # pixels.append(np.load(io.BytesIO(base64.b64decode(image.pixels))))
            pixels.append(image.pixels)
            points.append(image.point)
            datetimes.append(datetime.fromtimestamp(image.timestamp) if image.timestamp else None)

        model = SegmentorTraining.load_from_checkpoint(local_model_path)
        data_module = SegmentationDataModuleInference(
            gsd=data.images[0].gsd,
            bands=data.images[0].bands,
            pixels=pixels,
            platform=data.images[0].platform,
            wavelengths=data.images[0].wavelengths,
            points=points,
            datetimes=datetimes,
            batch_size=10,
            num_workers=0,
            train_test_ratio=0,
            labels=None,
        )
        data_module.setup()

        all_preds = []
        model.eval()

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

        all_preds = np.concatenate(all_preds, axis=0)

        return SegmentationLabels(labels=all_preds.tolist()[: len(pixels)])

    else:
        raise ValueError(f"Unknown model type: {data.model}. Supported models: classification, segmentation")
